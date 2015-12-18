require 'nn'
require 'nngraph'

-------------------------------------------------------------------------------
-- Grid LSTM core
-------------------------------------------------------------------------------

local layer, parent = torch.class('nn.GridLSTM', 'nn.Module')

function layer:__init(opt)
  parent.__init(self)

  -- options for core network
  self.seq_length = utils.getopt(opt, 'seq_length', 50) 
  self.num_layers = utils.getopt(opt, 'num_layers', 1)
  self.dim = utils.getopt(opt, 'dim', 2) -- dimensions in the N-LSTM
  self.rnn_size = utils.getopt(opt, 'rnn_size', 100)
  self.input_encoding_size = utils.getopt(opt, 'input_encoding_size')
  self.vocab_size = utils.getopt(opt, 'vocab_size')
  self.batch_size = utils.getopt(opt, 'batch_size')
  self.lookup_table = nn.LookupTable(self.vocab_size, self.rnn_size)
  
  self:_createInitState(self.batch_size)
end

-- Create two vector of zeros of size rnn_size corresponding
-- to initial memory and hidden states. Do this for each layer
-- and each dimension, resulting in (2 * layers * dim) zeroed vecs
function layer:_createInitState(batch_size)
  assert(batch_size ~= nil, 'batch size must be provided')
  -- construct the initial state for the LSTM
  if not self.init_state then self.init_state = {} end -- lazy init
 
  for h=1, 2*self.num_layers do
    -- note, the init state Must be zeros because we are using init_state to init grads in backward call too
    if self.init_state[h] then
      if self.init_state[h]:size(1) ~= batch_size then
        self.init_state[h]:resize(batch_size, self.rnn_size):zero() -- expand the memory
      end
    else
      self.init_state[h] = torch.zeros(batch_size, self.rnn_size)
    end
  end
  self.num_state = #self.init_state
end

function layer:createClones()
  -- construct the temporal clones
  print('constructing clones inside the GridLSTM layer')
  self.clones = {self.core}
  self.lookup_tables = {self.lookup_table}
  for t=2,self.seq_length+2 do
    self.clones[t] = self.core:clone('weight', 'bias', 'gradWeight', 'gradBias')
    self.lookup_tables[t] = self.lookup_table:clone('weight', 'gradWeight')
  end
end

--[[
input is a torch.LongTensor of size DxN, elements 1..M
   where M = opt.vocab_size, D = opt.input_seq_length, and N = opt.batch_size

returns a DxN torch.LongTensor giving softmax outputs.
--]]
function layer:updateOutput(input)
    -- clones is a sequence length set of clones of the grid-lstm core
    if self.clones == nil then self:createClones() end -- lazily create clones on first forward pass

    local batch_size = input:size(1)
    self.output:resize(self.seq_length, batch_size)

    -- set the initial memory and hidden states to zeros along the time dimension
    self.state = {[0] = self.init_state}
    self.inputs = {}
    self.lookup_tables_inputs = {}

    -- Iterate the sequence, collect predictions
    for t=1, self.seq_length do

    	-- Embed the input in memory and hidden space
      -- The paper says "A data point is projected into the network via a pair 
      -- of input hidden and memory vectors along one of the sides of the grid".
      -- I assume that means a lookup table for each? Zeros for the memory vector?
      -- Idk, can experiment a bit here.
      
      local input_c_d = torch.zeros(batch_size, self.rnn_size)
      
      self.lookup_tables_inputs[t] = input
      local inputs_h_d = self.lookup_tables[t]:forward(input) -- NxK sized input (token embedding vectors)
    	
    	-- construct the inputs
    	self.inputs[t] = {input_c_d, inputs_h_d, unpack(self.state[t-1])}
	    
	    -- forward the network
      local out = self.clones[t]:forward(self.inputs[t])

      -- process the outputs
      local depth_dim = 1
      local time_dim = 2
      self.output[t] = out[depth_dim][#out] -- last element is the prediction
      self.state[t] = {} -- the rest is state
      for i=1,self.num_state do table.insert(self.state[t], out[time_dim][i]) end
    end
    return self.output -- softmax outputs
end


--[[
gradOutput is an (D+2)xNx(M+1) Tensor.
--]]
function layer:updateGradInput(input, gradOutput)
end