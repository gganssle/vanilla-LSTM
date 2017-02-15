--[[
This script builds vectors for training and predicting.
]]--

require 'num_gen'

-- init
min = 1
max = 1000
len = 5
vpd = 111
ln = 'labels'
dn = 'data' 

-- files 1/2
lf = torch.DiskFile(ln, 'w')
df = torch.DiskFile(dn, 'w')

-- build vects and write out
for i = 1, 9 do
	-- calc
	for j = 1, vpd do
		one = increase(min, max, len, i)

		-- write out
		for k = 1, len do
			df:writeInt(one.vect[k])
			lf:writeInt(one.label[k])
		end
	end
end

-- files 2/2
lf:close()
df:close()
