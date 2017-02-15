--[[ 
This is a little toy LSTM which guesses the last digit of the
next number in a sequence.
NOTE: The first argument in `nn.LookupTable` must equal
the "max" value in your data, as in build_vects.lua
]]--

require 'rnn'

-- batch params
vnm = 900
bsz = 100
vlen = 5
epochs = 50
learningrate = 0.01
dn = 'data'
ln = 'labels'
outmod = '20170215.model'

-- build RNN --------------------------------------------
net = nn.Sequential()
net:add(nn.LookupTable(1000, 500))
net:add(nn.LSTM(500, 500, vlen))
net:add(nn.Dropout())
net:add(nn.Linear(500, 10))
net:add(nn.LogSoftMax())
net = nn.Sequencer(net)
---------------------------------------------------------

-- set training mode
net:training()

-- def criterion
criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())

-- send to CUDA
cuda = false
if pcall(require, 'cunn') and pcall(require, 'cutorch') then
	print('CUDA Torch found. Training on GPU')
	net = net:cuda()
	criterion = criterion:cuda()
	cuda = true
else
	print('No CUDA Torch. Training on CPU.')
end

-- timing 1/2
timer = torch.Timer()

	-- count
	count = 0

-- files 1/2
df = torch.DiskFile(dn, 'r')
lf = torch.DiskFile(ln, 'r')

-- batch loop
for batch = 1, vnm, bsz do
	-- count
	count = count + 1

	-- build data
	trndat = nil
	trnlbl = nil
	collectgarbage()
	trndat = torch.Tensor(bsz, vlen)
	trnlbl = torch.Tensor(bsz, vlen)

	-- load data
	for vect = 1, bsz do
		for ftr = 1, vlen do
			trndat[vect][ftr] = df:readInt()
			trnlbl[vect][ftr] = lf:readInt()
		end
	end

	--if batch == 1 then print(trndat) print(trnlbl) end

	-- send to CUDA
	if cuda then
		trndat = trndat:cuda()
		trnlbl = trnlbl:cuda()
	end

	-- train
	print('batch # = ' .. count .. '/' .. vnm/bsz)
	for trnidx = 1, epochs do
		-- update params 1/2
		net:zeroGradParameters()

		-- Forward pass
		local netOut = net:forward(trndat)
		local err = criterion:forward(netOut, trnlbl)

		--progress
		if trnidx % 1 == 0 then
			print(string.format("Iteration %d ; NLL err = %f ", trnidx, err))
		end

		-- Backward pass
		local gradOutput = criterion:backward(netOut, trnlbl)
		net:backward(trndat, gradOutput)

		-- update params 2/2
		net:updateParameters(learningrate)
	end

end

-- save network weights + biases
torch.save(outmod, net)

-- timing 2/2
print('training time = ' .. timer:time().real .. ' seconds')

-- set evaluation mode
net:evaluate()

-- predictions on trained vector
rand = torch.random(1, vnm)
print('Predicting on a random learned vector of index ', rand)
new = torch.Tensor(1,vlen)
for i = 1, new:size(2) do
	if rand % bsz == 0 then
		temp = 1
	else
		temp = rand % bsz +1
	end
	new[1][i] = trndat[temp][i]
end
print(new)
--print(net:forward(new):exp())
conf, idx = torch.sort(net:forward(new), true)
print('prediction is: ', idx[{{},{},{1}}])

-- real predictions on unseen vector
for i = 1, vlen do
	new[1][i] = df:readInt()
end
print('predicting on this (unseen) vector:')
print(new)
--print(net:forward(new):exp())
conf, idx = torch.sort(net:forward(new), true)
print('prediction is', idx[{{},{},{1}}])

-- files 2/2
df:close()
lf:close()


