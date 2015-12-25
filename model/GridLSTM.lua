require 'nn'
require 'nngraph'
--[[
  Takes h_t and h_d, hidden states from the temporal and 
  depth dimensions respectively, as well as prev_c, the 
  dimension's previous memory cell.

  Computes next_c, next_h along the dimension, using the standard
  lstm gated update, conditioned on the concatenated time and 
  depth hidden states.
--]]
function lstm(h_t, h_d, prev_c, rnn_size)
  local all_input_sums = nn.CAddTable()({h_t, h_d})
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
  --]]
local GridLSTM2 = {}
function GridLSTM2.grid_lstm(input_size, rnn_size, n, dropout, should_tie_weights)
  dropout = dropout or 0 

  -- There will be 2*n+1 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- input c for depth dimension
  table.insert(inputs, nn.Identity()()) -- input h for depth dimension
  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prev_c[L] for time dimension
    table.insert(inputs, nn.Identity()()) -- prev_h[L] for time dimension
  end

  local shared_weights
  if should_tie_weights == 1 then shared_weights = {nn.Linear(rnn_size, 4 * rnn_size), nn.Linear(rnn_size, 4 * rnn_size)} end

  local outputs_t = {} -- Outputs being handed to the next time step along the time dimension
  local outputs_d = {} -- Outputs being handed from one layer to the next along the depth dimension

  for L = 1,n do
    -- Take hidden and memory cell from previous time steps
    local prev_c_t = inputs[L*2+1]
    local prev_h_t = inputs[L*2+2]

    if L == 1 then
      -- In first layer
      prev_c_d = inputs[1] -- input_c_d: just zeros
      prev_h_d = nn.LookupTable(input_size, rnn_size)(inputs[2]) -- input_h_d: map point in vocab space into hidden space w/ a lookup table
    else 
      -- In layers 2...N
      -- Take hidden and memory cell from layers below
      prev_c_d = outputs_d[((L-1)*2)-1]
      prev_h_d = outputs_d[((L-1)*2)]
      if dropout > 0 then prev_h_d = nn.Dropout(dropout)(prev_h_d):annotate{name='drop_' .. L} end -- apply dropout, if any
    end

    -- Evaluate the input sums at once for efficiency
    local t2h_t = nn.Linear(rnn_size, 4 * rnn_size)(prev_h_t):annotate{name='i2h_'..L}
    local d2h_t = nn.Linear(rnn_size, 4 * rnn_size)(prev_h_d):annotate{name='h2h_'..L}
    
    -- Get transformed memory and hidden states pointing in the time direction
    local next_c_t, next_h_t = lstm(t2h_t, d2h_t, prev_c_t, rnn_size)

    -- Pass memory cell and hidden state to next timestep
    table.insert(outputs_t, next_c_t)
    table.insert(outputs_t, next_h_t)

    -- Evaluate the input sums at once for efficiency
    local t2h_d = nn.Linear(rnn_size, 4 * rnn_size)(next_h_t):annotate{name='i2h_'..L}
    local d2h_d = nn.Linear(rnn_size, 4 * rnn_size)(prev_h_d):annotate{name='h2h_'..L}

    if tie_weights == 1 then
      t2h_d.data.module:share(shared_weights[1], 'weight', 'bias', 'gradWeight', 'gradBias')
      d2h_d.data.module:share(shared_weights[2], 'weight', 'bias', 'gradWeight', 'gradBias')
    end
    
    -- Create lstm gated update pointing in the depth direction
    -- Prioritize the depth dimension by using the modulated temporal hidden state as input
    -- instead of the previous temporal hidden state. See section 3.2 of this
    -- for an explanation http://arxiv.org/pdf/1507.01526v2.pdf 
    local next_c_d, next_h_d = lstm(t2h_d, d2h_d, prev_c_d, rnn_size)

    -- Pass memory cell and hidden state to layer above
    table.insert(outputs_d, next_c_d)
    table.insert(outputs_d, next_h_d)
  end

  -- set up the decoder
  local top_h = outputs_d[#outputs_d]
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end
  local proj = nn.Linear(rnn_size, input_size)(top_h):annotate{name='decoder'}
  local logsoft = nn.LogSoftMax()(proj)
  table.insert(outputs_t, logsoft)

  return nn.gModule(inputs, outputs_t)
end

return GridLSTM2

