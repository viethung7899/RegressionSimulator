/// <reference path="../libraries/p5.d/p5.global-mode.d.ts" />

// data points
let X = [];
let Y = [];

// parameters for machine learning
const a = tf.scalar(Math.random(1)).variable();
const b = tf.scalar(Math.random(1)).variable();

// Stochastic Gradient Descent
const learningRate = 0.5;
const optimizer = tf.train.sgd(learningRate);

// y = f(x) = ax + b
const linear = (x) => a.mul(x).add(b);
const quadratic = 0
const cubic = 0;

// root mean square erroe
const loss = (pred, label) => pred.sub(label).square().mean();

const train = (tensorX, tensorY) => {
  optimizer.minimize(() => loss(linear(tensorX), tensorY));
};

// Drawing
function setup() {
  createCanvas(windowWidth, windowHeight);
}

function draw() {
  background(51);
  showPoints();

  // Train the model
  if (X.length) {
    tf.tidy(() => {
      const tensorX = tf.tensor1d(X);
      const tensorY = tf.tensor1d(Y);

      train(tensorX, tensorY);
    });

    drawLine();
  }

  // console.log(tf.memory().numTensors);
}

// Click the points
function mouseClicked() {
  const normX = map(mouseX, 0, windowWidth, 0, 1);
  const normY = map(mouseY, 0, windowHeight, 0, 1);
  X.push(normX);
  Y.push(normY);
}

// Show points
function showPoints() {
  noStroke();
  fill(255);
  for (let i = 0; i < X.length; i++) {
    const x = Math.floor(map(X[i], 0, 1, 0, windowWidth));
    const y = Math.floor(map(Y[i], 0, 1, 0, windowHeight));
    ellipse(x, y, 10);
  }

  noFill();
}

// Draw line
function drawLine() {
  const x = tf.tensor1d([0, 1]);
  const y = tf.tidy(() => linear(x));

  const xs = x.dataSync();
  const ys = y.dataSync();

  x.dispose();
  y.dispose();

  const x0 = map(xs[0], 0, 1, 0, windowWidth);
  const y0 = map(ys[0], 0, 1, 0, windowHeight);
  const x1 = map(xs[1], 0, 1, 0, windowWidth);
  const y1 = map(ys[1], 0, 1, 0, windowHeight);

  stroke(255);
  strokeWeight(2);
  line(x0, y0, x1, y1);
}

