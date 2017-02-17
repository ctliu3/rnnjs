String.prototype.format = function() {
  var formatted = this;
  for( var arg in arguments ) {
    formatted = formatted.replace("{" + arg + "}", arguments[arg]);
  }
  return formatted;
};

function assert(condition, msg) {
  if (!condition) {
    msg = msg || "Assertion failed";
    throw Error(msg);
  }
}

function FilledArray(n, value) {
  var arr = new Array(n);
  for (var i = 0; i < n; ++i) {
    arr[i] = value;
  }
  return arr;
}

function Mat(n, d) {
  this.n = n;
  this.d = d;
  this.w = FilledArray(n * d, 0);
  this.dw = FilledArray(n * d, 0);
}

function RandMat(n, d, mu, std) {
  var m = new Mat(n, d);
  for (var i = 0; i < m.w.length; ++i) {
    m.w[i] = rand(-std, std);
  }
  return m;
}

function rand(a, b) {
  return Math.random() * (b - a) + a;
}

exports.Assert = assert;
exports.FilledArray = FilledArray;
exports.Mat = Mat;
exports.RandMat = RandMat;
