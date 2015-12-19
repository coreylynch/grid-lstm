require 'nn'
require 'nngraph'

--[[
Inputs:
  prev_h_t: previous hidden state along the temporal dimension
  prev_h_d: previous hidden state along the depth dimension
  rnn_size: size of the internal state vectors
  shared_weights: reference weights for all layers to share if we're doing weight tying.
Returns:
  next_c: the transformed memory cell
  next_h: the transformed hidden state
]]--
function lstm(prev_h_t, prev_h_d, rnn_size, shared_weights)
  -- evaluate the input sums at once for efficiency
  local t2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h_t):annotate{name='h2h_'..L}
  local d2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h_d):annotate{name='i2h_'..L}
  
  -- if shared weights is set, tie all the transform weights and gradients
  if shared_weights then
    t2h.data.module:share(shared_weights[1].data.module, 'weight', 'bias', 'gradWeight', 'gradBias')
    d2h.data.module:share(shared_weights[2].data.module, 'weight', 'bias', 'gradWeight', 'gradBias')
  end

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

  output_size: vocab_size
  rnn_size: size of every internal state vector
  n: number of layers

--]]
local GridLSTM = {}
function GridLSTM.grid_lstm(input_size, output_size, rnn_size, n, dropout, tie_weights)
  dropout = dropout or 0 

  -- subscript t refers to time dimension, subscript d refers to depth dimension vv
  --
  -- Layer 1
  -- receive input mapped to (m_x, h_x)
  -- H = [h_x, prev_h_t]
  -- m_t_next, h_t_next = LSTM_t(H)
  -- H_prime = (h_x, h_t_next)
  -- m_d_next, h_d_next = LSTM_d(H_prime)
  -- outputs = {m_d_next, h_d_next, m_t_next, h_t_next}

  -- Layer 2...L
  -- H = [prev_h_d, prev_h_t]
  -- m_t_next, h_t_next = LSTM_t(H)
  -- H_prime = (h_x, h_t_next)
  -- m_d_next, h_d_next = LSTM_d(H_prime)
  -- outputs = {m_d_next, h_d_next, m_t_next, h_t_next}

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

  table.insert(inputs, nn.Identity()()) -- input c for depth dimension
  table.insert(inputs, nn.Identity()()) -- input h for depth dimension

  -- Set up prev hidden inputs for each layer
  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prev_c[L] for time dimension
    table.insert(inputs, nn.Identity()()) -- prev_h[L] for time dimension
  end

  local shared_weights

  -- Create a table of outputs of size #dim
  local outputs_t = {} -- outputs being handed to the next time step along the time dimension
  local outputs_d = {} -- outputs being handed from one layer to the next along the depth dimension

  -- from bottom layer to top layer,
  -- map inputs to hidden layer,
  for L = 1,n do

    -- take prev_c_t, prev_h_t from inputs
    local prev_c_t = inputs[L*2+1]
    local prev_h_t = inputs[L*2+2]

    -- if in first layer, take prev_c_d, prev_h_d from inputs
    -- else if in layers 2...N, take them from layer below
    local prev_c_d
    local prev_h_d
    if L == 1 then
      -- in first layer
      prev_c_d = inputs[1] -- input_c_d: just zeros?
      -- prev_h_d = inputs[2] -- input_h_d
      prev_h_d = nn.LookupTable(input_size, rnn_size)(inputs[2]) -- input_h_d: map point in vocab space into hidden space w/ a lookup table

      -- If we're tying weights along the depth dimension (as suggested in the paper),
      -- create reference weights in the first layer and hand them in to all subsequent
      -- layers to be shared.
      if tie_weights == 1 then shared_weights = {nn.Linear(rnn_size, 4 * rnn_size), nn.Linear(rnn_size, 4 * rnn_size)} end
    else
      -- in layers 2...N
      prev_c_d = outputs_d[((L-1)*2)-1]
      prev_h_d = outputs_d[((L-1)*2)]
      if dropout > 0 then prev_h_d = nn.Dropout(dropout)(prev_h_d):annotate{name='drop_' .. L} end -- apply dropout, if any
    end

    -- Do a forward pass on the LSTM pointing in the time direction
    local next_c_t, next_h_t = lstm(prev_h_t, prev_h_d, rnn_size)

    -- Do a forward pass on LSTM pointing in the depth direction (towards output)
    -- Prioritize the depth dimension by using the modulated temporal hidden state as input
    -- instead of the previous temporal hidden state. See section 3.2 of this
    -- for an explanation http://arxiv.org/pdf/1507.01526v2.pdf    
    local next_c_d, next_h_d = lstm(next_h_t, prev_h_d, rnn_size, shared_weights) 

    table.insert(outputs_d, next_c_d)
    table.insert(outputs_d, next_h_d)
    table.insert(outputs_t, next_c_t)
    table.insert(outputs_t, next_h_t)
  end

  -- set up the decoder
  local top_h = outputs_d[#outputs_d]
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h):annotate{name='drop_final'} end
  local proj = nn.Linear(rnn_size, output_size)(top_h):annotate{name='decoder'}
  local logsoft = nn.LogSoftMax()(proj)
  table.insert(outputs_t, logsoft)

  -- for N layers, you'll have 2*N+1 outputs to hand to the next timestep,
  -- (a hidden and memory cell for each layer + the prediction)
  return nn.gModule(inputs, outputs_t)
end

return GridLSTM