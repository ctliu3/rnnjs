var util = require('./util.js');

// exports.LSTMNet = function(input_size, hidden_sizes, output_size) {
function Graph(needs_backprop) {
  if (typeof needs_backprop == 'undefined') {
    needs_backprop = true;
  }
  this.needs_backprop = needs_backprop;
  this.backprop = [];
}

Graph.prototype = {
  backward: function() {
    for (var i = this.backprop.length - 1; i >= 0; --i) {
      this.backprop[i]();
    }
  },

  tanh: function(m) {
    var ret = new util.Mat(m.n, m.d);
    for (var i = 0; i < m.w.length; ++i) {
      ret.w[i] = Math.tanh(m.w[i]);
    }

    if (this.needs_backprop) {
      var backward = function() {
        for (var i = 0; i < m.w.length; ++i) {
          m.dw[i] += (1. - m.w[i] * m.w[i]) * ret.dw[i];
        }
      }
      this.backprop.push(backward);
    }
    return ret;
  },

  sigmoid: function(m) {
    var ret = new util.Mat(m.n, m.d);
    for (var i = 0; i < m.w.length; ++i) {
      ret.w[i] = 1. / (1 + Math.exp(m.w[i]));
    }
    if (this.needs_backprop) {
      var backward = function() {
        for (var i = 0; i < m.w.length; ++i) {
          m.dw[i] += m.w[i] * (1. - m.w[i]) * ret.dw[i];
        }
      }
      this.backprop.push(backward);
    }
    return ret;
  },

  add: function(m1, m2) {
    util.Assert(m1.w.length == m2.w.length, "func add(): {0} != {1}".format(m1.w.length, m2.w.length));

    var ret = new util.Mat(m1.n, m1.d);
    for (var i = 0; i < m1.w.length; ++i) {
      ret.w[i] = m1.w[i] + m2.w[i];
    }
    if (this.needs_backprop) {
      var backward = function() {
        for (var i = 0; i < m1.w.length; ++i) {
          m1.dw[i] += ret.dw[i];
          m2.dw[i] += ret.dw[i];
        }
      }
      this.backprop.push(backward);
    }
    return ret;
  },

  mul: function(m1, m2) {
    util.Assert(m1.d == m2.n, "func mul(): m1.d != m2.n, {0} != {1}".format(m1.d, m2.n));

    var ret = new util.Mat(m1.n, m2.d);
    for (var i = 0; i < m1.n; ++i) {
      for (var j = 0; j < m2.d; ++j) {
        var dot = 0;
        for (var k = 0; k < m1.d; ++k) {
          dot += m1.w[i * m1.d + k] * m2.w[k * m2.d + j];
        }
        ret.w[i * m2.d + j] = dot;
      }
    }
    if (this.needs_backprop) {
      var backward = function() {
        for (var i = 0; i < m1.n; ++i) {
          for (var j = 0; j < m2.d; ++j) {
            for (var k = 0; k < m1.d; ++k) {
              m1.dw[i * m1.d + k] += m2.w[k * m2.d + j] * m.dw[i * m1.d + j];
              m2.dw[k * m2.d + j] += m1.w[i * m1.d + k] * m.dw[i * m1.d + j];
            }
          }
        }
      }
      this.backprop.push(backward);
    }

    return ret;
  },

  eltmul: function(m1, m2) {
    util.Assert(m1.w.length == m2.w.length, "func eltmul(): {0} != {1}".format(m1.w.length, m2.w.length));

    var ret = new util.Mat(m1.n, m1.d);
    for (var i = 0; i < m1.w.length; ++i) {
      ret.w[i] = m1.w[i] * m2.w[i];
    }
    if (this.needs_backprop) {
      var backward = function() {
        for (var i = 0; i < m1.w.length; ++i) {
          m1.dw[i] += m2.w[i] * ret.dw[i];
          m2.dw[i] += m1.w[i] * ret.dw[i];
        }
      }
      this.backprop.push(backward);
    }
    return ret;
  }
}

function LSTMNet(input_size, hidden_sizes, output_size) {
  // hidden_size is the number of output in the hidden layer,
  // it should be a list

  model = {};
  for (var d = 0; d < hidden_sizes.length; ++d) {
    var prev_size = d == 0 ? input_size : hidden_sizes[d - 1];
    var hidden_size = hidden_sizes[d];

    // input gate
    model['Wix' + d] = new util.RandMat(hidden_size, prev_size, 0, 0.08);
    model['Wih' + d] = new util.RandMat(hidden_size, hidden_size, 0, 0.08);
    model['bi' + d] = new util.Mat(hidden_size, 1);

    // forget gate
    model['Wfx' + d] = new util.RandMat(hidden_size, prev_size, 0, 0.08);
    model['Wfh' + d] = new util.RandMat(hidden_size, hidden_size, 0, 0.08);
    model['bf' + d] = new util.Mat(hidden_size, 1);

    // output gate
    model['Wox' + d] = new util.RandMat(hidden_size, prev_size, 0, 0.08);
    model['Woh' + d] = new util.RandMat(hidden_size, hidden_size, 0, 0.08);
    model['bo' + d] = new util.Mat(hidden_size, 1);

    // cell
    model['Wcx' + d] = new util.RandMat(hidden_size, prev_size, 0, 0.08);
    model['Wch' + d] = new util.RandMat(hidden_size, hidden_size, 0, 0.08);
    model['bc' + d] = new util.Mat(hidden_size, 1);
  }

  // for final output layer?
  model['Whd'] = new util.RandMat(output_size, hidden_size, 0, 0.08);
  model['bd'] = new util.Mat(output_size, 1);

  this.model = model;
}

LSTMNet.prototype.forward = function(G, hidden_sizes, x, prev) {
  if (typeof prev.h === 'undefined') {
    var hidden_prevs = [];
    var cell_prevs = [];
    for (var d = 0; d < hidden_sizes.length; ++d) {
      hidden_prevs.push(new util.Mat(hidden_sizes[d], 1));
      cell_prevs.push(new util.Mat(hidden_sizes[d], 1));
    }
  } else {
    var hidden_prevs = prev.h;
    var cell_prevs = prev.c;
  }

  var hidden = [];
  var cell = [];
  for (var d = 0; d < hidden_sizes.length; ++d) { // d means depth
    var curr_input = d == 0 ? x : hidden[d - 1];
    var hidden_prev = hidden_prevs[d];
    var cell_prev = cell_prevs[d];

    // input gate
    var h0 = G.mul(this.model['Wix' + d], curr_input);
    var h1 = G.mul(this.model['Wih' + d], hidden_prev);
    var input_gate = G.sigmoid(G.add(G.add(h0, h1), this.model['bi' + d]));

    // forget gate
    var h2 = G.mul(this.model['Wfx' + d], curr_input);
    var h3 = G.mul(this.model['Wfh' + d], hidden_prev);
    var forget_gate = G.sigmoid(G.add(G.add(h2, h3), this.model['bf' + d]));

    // output gate
    var h4 = G.mul(this.model['Wox' + d], curr_input);
    var h5 = G.mul(this.model['Woh' + d], hidden_prev);
    var output_gate = G.sigmoid(G.add(G.add(h4, h5), this.model['bo' + d]));

    // cell hypothesis
    var h6 = G.mul(this.model['Wcx' + d], curr_input);
    var h7 = G.mul(this.model['Wch' + d], hidden_prev);
    var cell_hypoth = G.tanh(G.add(G.add(h6, h7), this.model['bo' + d]));

    // cell output
    var keep_cell = G.eltmul(forget_gate, cell_prev);
    var write_cell = G.eltmul(input_gate, cell_hypoth);
    var cell_d = G.add(keep_cell, write_cell);

    // output in this cell
    var hidden_d = G.eltmul(output_gate, G.tanh(cell_d));

    hidden.push(hidden_d);
    cell.push(cell_d);
  }

  var output = G.add(G.mul(this.model['Whd'], hidden[hidden.length - 1]), this.model['bd']);

  return {'h': hidden, 'o': output, 'c': cell};
}


exports.LSTMNet = LSTMNet;
exports.Graph = Graph;
