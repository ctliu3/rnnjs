var lstm = require('./lstm.js');
var util = require('./util.js');

var hidden_sizes = [10, 10];
var net = new lstm.LSTMNet(5, hidden_sizes, 2);
// console.log(net);

var G = new lstm.Graph(true);
var x1 = new util.Mat(5, 1);
var x2 = new util.Mat(5, 1);
var x3 = new util.Mat(5, 1);

var out1 = net.forward(G, hidden_sizes, x1, {});
var out2 = net.forward(G, hidden_sizes, x1, out1);
var out3 = net.forward(G, hidden_sizes, x1, out2);

// G.backward();
