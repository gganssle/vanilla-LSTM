--[[
This script generates sequences of integers in which the
last digits of the integers is monotonically increasing
by one. Here's an example:

	51, 2, 7893, 334, 825

It also generates the next increased digit (for use as
a training label). E.g.

	6

in the case of the above example.
]]--

function increase(min, max, len, digit)
	-- init
	local out = {}
	local poss = torch.Tensor{1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9}
	local dict = {}

	for i = 1, 9 do
		table.insert(dict, {})
	end

	-- generate lookup table
	for i = min, max do
		for j = 1, 10 do
			if i % 10 == j then
				table.insert(dict[j], i)
			end
		end
	end

	-- pick vector numbers
	local vect = torch.Tensor(len)
	for i = 1, len do
		idx = poss[digit + 9 - len + i]
		vect[i] = dict[idx][torch.random(1,table.getn(dict[idx]))]
	end

	-- get label
	local label = torch.Tensor(len)
	for i = 1, len do
		label[i] = poss[digit + 9 + 1 - len + i]
	end

	-- output
	out.vect = vect
	out.label = label
	return out
end


