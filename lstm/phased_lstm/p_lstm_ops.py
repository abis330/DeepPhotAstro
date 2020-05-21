from tensorflow.contrib.rnn import MultiRNNCell
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest


class MultiPRNNCell(MultiRNNCell):
  """ Phased RNN cell composed sequentially of multiple simple phased cells.
  For used with cells that take input on the form (time, x), like
  tf.contrib.rnn.PhasedLSTMCell
  """

  def call(self, inputs, state):
    """Run this multi-layer cell on inputs, starting from state."""
    cur_state_pos = 0
    (time, cur_inp) = inputs
    new_states = []
    for i, cell in enumerate(self._cells):
      with vs.variable_scope("cell_{}".format(i)):
        if self._state_is_tuple:
          if not nest.is_sequence(state):
            raise ValueError(
              ("Expected state to be a tuple of length {}, "
               "but received: {}".format(
                len(self.state_size), state)
              )
            )
          cur_state = state[i]
        else:
          cur_state = array_ops.slice(state, [0, cur_state_pos],
                                      [-1, cell.state_size])
          cur_state_pos += cell.state_size
        cur_inp, new_state = cell((time, cur_inp), cur_state)
        new_states.append(new_state)

    new_states = (tuple(new_states) if self._state_is_tuple else
                  array_ops.concat(new_states, 1))

    return cur_inp, new_states
