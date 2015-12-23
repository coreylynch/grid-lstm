require 'nn'
require 'nngraph'

local GridLSTM2 = {}
function GridLSTM2.grid_lstm(input_size, rnn_size, n, dropout, tie_weights)
  dropout = dropout or 0 

  -- there will be 2*n+1 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- input c for depth dimension
  table.insert(inputs, nn.Identity()()) -- input h for depth dimension
  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prev_c[L] for time dimension
    table.insert(inputs, nn.Identity()()) -- prev_h[L] for time dimension
  end

  local outputs_t = {} -- outputs being handed to the next time step along the time dimension
  local outputs_d = {} -- outputs being handed from one layer to the next along the depth dimension
  for L = 1,n do
    -- c,h from previos timesteps
    local prev_c_t = inputs[L*2+1]
    local prev_h_t = inputs[L*2+2]
    -- the input to this layer
    if L == 1 then
      prev_c_d = inputs[1] -- input_c_d: just zeros?
      prev_h_d = nn.LookupTable(input_size, rnn_size)(inputs[2]) -- input_h_d: map point in vocab space into hidden space w/ a lookup table
    else 
      prev_c_d = outputs_d[((L-1)*2)-1]
      prev_h_d = outputs_d[((L-1)*2)]
      if dropout > 0 then prev_h_d = nn.Dropout(dropout)(prev_h_d):annotate{name='drop_' .. L} end -- apply dropout, if any
    end
    -- evaluate the input sums at once for efficiency
    local t2h_t = nn.Linear(rnn_size, 4 * rnn_size)(prev_h_t):annotate{name='i2h_'..L}
    local d2h_t = nn.Linear(rnn_size, 4 * rnn_size)(prev_h_d):annotate{name='h2h_'..L}
    local all_input_sums_t = nn.CAddTable()({t2h_t, d2h_t})

    local reshaped_t = nn.Reshape(4, rnn_size)(all_input_sums_t)
    local n1_t, n2_t, n3_t, n4_t = nn.SplitTable(2)(reshaped_t):split(4)
    -- decode the gates
    local in_gate_t = nn.Sigmoid()(n1_t)
    local forget_gate_t = nn.Sigmoid()(n2_t)
    local out_gate_t = nn.Sigmoid()(n3_t)
    -- decode the write inputs
    local in_transform_t = nn.Tanh()(n4_t)
    -- perform the LSTM update
    local next_c_t           = nn.CAddTable()({
        nn.CMulTable()({forget_gate_t, prev_c_t}),
        nn.CMulTable()({in_gate_t,     in_transform_t})
      })
    -- gated cells form the output
    local next_h_t = nn.CMulTable()({out_gate_t, nn.Tanh()(next_c_t)})
    
    table.insert(outputs_t, next_c_t)
    table.insert(outputs_t, next_h_t)


    -- evaluate the input sums at once for efficiency
    local t2h_d = nn.Linear(rnn_size, 4 * rnn_size)(next_h_t):annotate{name='i2h_'..L}
    local d2h_d = nn.Linear(rnn_size, 4 * rnn_size)(prev_h_d):annotate{name='h2h_'..L}
    local all_input_sums_d = nn.CAddTable()({t2h_d, d2h_d})

    local reshaped_d = nn.Reshape(4, rnn_size)(all_input_sums_d)
    local n1_d, n2_d, n3_d, n4_d = nn.SplitTable(2)(reshaped_d):split(4)
    -- decode the gates
    local in_gate_d = nn.Sigmoid()(n1_d)
    local forget_gate_d = nn.Sigmoid()(n2_d)
    local out_gate_d = nn.Sigmoid()(n3_d)
    -- decode the write inputs
    local in_transform_d = nn.Tanh()(n4_d)
    -- perform the LSTM update
    local next_c_d           = nn.CAddTable()({
        nn.CMulTable()({forget_gate_d, prev_c_d}),
        nn.CMulTable()({in_gate_d,     in_transform_d})
      })
    -- gated cells form the output
    local next_h_d = nn.CMulTable()({out_gate_d, nn.Tanh()(next_c_d)})

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

