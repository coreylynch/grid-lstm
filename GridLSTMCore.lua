require 'nn'
require 'nngraph'
require 'LSTM'

--[[
Inputs:
  prev_h_t: previous hidden state along the temporal dimension
  prev_h_d: previous hidden state along the depth dimension
  rnn_size: size of the internal state vectors
Returns:
  next_c: the transformed memory cell
  next_h: the transformed hidden state
]]--
function lstm(prev_h_t, prev_h_d, rnn_size)
  -- evaluate the input sums at once for efficiency
  local t2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h_t):annotate{name='i2h_'..L}
  local d2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h_d):annotate{name='h2h_'..L}
  local all_input_sums = nn.CAddTable()({t2h, d2h})

  local reshaped = nn.Reshape(4, rnn_size)(all_input_sums)
  local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
  -- decode the gates
  local in_gate = nn.Sigmoid()(n1)
  local forget_gate = nn.Sigmoid()(n2)
  local out_gate = nn.Sigmoid()(n3)
  -- decode the write inputs
  local in_transform = nn.Tanh()(n4)
  -- perform the LSTM update
  local next_c           = nn.CAddTable()({
      nn.CMulTable()({forget_gate, prev_c}),
      nn.CMulTable()({in_gate,     in_transform})
    })
  -- gated cells form the output
  local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
  return next_c, next_h
end

--[[
GridLSTM:
  1) Map input x into memory and hidden cells m(1), h(1) along the depth dimension.
  2) Concatenate hidden cells from time and depth dimensions, [h(1), h(2)] into H.
  3) Forward the time LSTM, LSTM_2(H) -> h(2)', m(2)'; this is following 3.2 "Priority dimensions"
  4) Concatenate transformed h(2)' and h(1) into H' = [h(1), h(2)']
  5) Forward the depth LSTM, LSTM_1(H') -> h(1)', m(1)'
  6) Either repeat 1-5 for another layer or read a prediction off of h(1)', m(1)'

  input_size: input embedding size
  output_size: vocab_size
  rnn_size: size of every internal state vector
  n: number of layers

--]]
local GridLSTM = {}
function GridLSTM.grid_lstm(input_size, output_size, rnn_size, n, dropout)
  dropout = dropout or 0 


  local x = nn.Identity()()
  local m_x = nn.Linear(input_size, rnn_size)(x)
  local h_x = nn.Linear(input_size, rnn_size)(x)

  -- Layer 1
  -- input is x, mapped to (m_x, h_x)
  -- H = [h_x, prev_h_2]
  -- m_2_next, h_2_next = LSTM_2(H)
  -- H_prime = (h_x, h_2_next)
  -- m_1_next, h_1_next = LSTM_1(H_prime)
  -- outputs = {m_1_next, h_1_next, m_2_next, h_2_next}
  

  -- Layer 2...L
  -- H = [prev_h_1, prev_h_2]
  -- m_2_next, h_2_next = LSTM_2(H)
  -- H_prime = (h_x, h_2_next)
  -- m_1_next, h_1_next = LSTM_1(H_prime)
  -- outputs = {m_1_next, h_1_next, m_2_next, h_2_next}

  -- There will be 2*n*d inputs
  -- Note that prev_c and prev_h for layer 1, dimension 1 is just the input projected into the respective vectors
  -- local inputs = {}
  -- for L = 1,n do
  --   for d = 1,dim do
  --     table.insert(inputs, nn.Identity()()) -- prev_c[L] for dim d
  --     table.insert(inputs, nn.Identity()()) -- prev_h[L] for dim d
  --   end
  -- end

  -- There will be 2*n*d inputs, or 2*n indexed by dimension
  -- Note that prev_c and prev_h for layer 1, dimension 1 is just the input projected into the respective vectors
  local inputs = {}
  for d=1,dim do
    inputs[d]={}
  end
  for L = 1,n do
    for d = 1,dim do
      table.insert(inputs[d], nn.Identity()()) -- prev_c[L] for dim d
      table.insert(inputs[d], nn.Identity()()) -- prev_h[L] for dim d
    end
  end

  local x, input_size_L
  local outputs = {}

  local depth_dim = 1
  local time_dim = 2

  -- from bottom layer to top layer,
  -- map inputs to hidden layer,
  for L = 1,n do

    -- take prev_c_t, prev_h_t from inputs
    local prev_c_t = inputs[time_dim][L*2-1]
    local prev_h_t = inputs[time_dim][L*2]

    -- if in first layer, take prev_c_d, prev_h_d from inputs
    -- else if in layers 2...N, take them from layer below
    local prev_c_d
    local prev_h_d
    if L == 1 then
      -- in first layer
      prev_c_d = inputs[depth_dim][L*2-1]
      prev_h_d = inputs[depth_dim][L*2]
    else
      -- in layers 2...N
      prev_c_d = outputs[depth_dim][((L-1)*2)-1]
      prev_h_d = outputs[depth_dim][((L-1)*2)]
      if dropout > 0 then prev_h_d = nn.Dropout(dropout)(prev_h_d):annotate{name='drop_' .. L} end -- apply dropout, if any
    end

    -- H by concatenating [prev_h_d, prev_h_t]
    local H = nn.JoinTable(1)({prev_h_d, prev_h_t})

    -- Do a forward pass on LSTM_t, giving next temporal memory cell and next temporal hidden state,
    -- e.g. c_t_next, h_t_next = LSTM_t(H)
    local next_c_t, next_h_t = lstm(prev_h_t, prev_h_d, rnn_size)

    -- Prioritize the depth dimension by using the modulated temporal hidden state as input
    -- instead of the previous temporal hidden state. See section 3.2 of this
    -- for an explanation http://arxiv.org/pdf/1507.01526v2.pdf
    local next_c_d, next_h_d = lstm(next_h_t, prev_h_d, rnn_size) 
    
    -- H_prime = (h_x, h_2_next)
    -- m_1_next, h_1_next = LSTM_1(H_prime)
    -- outputs = {m_1_next, h_1_next, m_2_next, h_2_next}


    table.insert(outputs, next_c)
    table.insert(outputs, next_h)
  end

  -- set up the decoder
  local top_h = outputs[#outputs]
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h):annotate{name='drop_final'} end
  local proj = nn.Linear(rnn_size, output_size)(top_h):annotate{name='decoder'}
  local logsoft = nn.LogSoftMax()(proj)
  table.insert(outputs, logsoft)

  return nn.gModule(inputs, outputs)
end

return LSTM

