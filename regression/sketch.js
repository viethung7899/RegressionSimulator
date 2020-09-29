/// <reference path="../libraries/p5.d/p5.global-mode.d.ts" />

// data points
let X = [];
let Y = [];

// Plotting X
const plotX = [];
for (let x = 0; x <= 1; x += 0.01) {
  plotX.push(x);
}
const tensorPlotX = tf.tensor1d(plotX);

// parameters for machine learning
let a, b, c, d;

// Stochastic Gradient Descent
const learningRate = 0.1;
const optimizer = tf.train.adam(learningRate);

// y = f(x) = ax + b
const functions = {
  linear: (x) => a.mul(x).add(b),
  quadratic: (x) => a.mul(tf.square(x)).mul(b.mul(x)).add(c),
  cubic: (x) =>
    a
      .mul(tf.pow(x, tf.scalar(3)))
      .add(b.mul(tf.pow(x, [2])))
      .add(c.mul(x))
      .add(d),
};

let predict = functions.linear;

// root mean square erroe
const loss = (pred, label) => pred.sub(label).square().mean();

const train = (tensorX, tensorY) => {
  optimizer.minimize(() => loss(predict(tensorX), tensorY));
};

// Drawing
function setup() {
  const canvas = createCanvas(windowWidth, windowHeight);
  canvas.mouseClicked(addPoint);
  resetCanvas();
}

function draw() {
  background(42);
  showPoints();

  // Train the model
  if (X.length) {
    tf.tidy(() => {
      const tensorX = tf.tensor1d(X);
      const tensorY = tf.tensor1d(Y);

      train(tensorX, tensorY);
    });

    drawCurve();
  }

  // console.log(tf.memory().numTensors);
}

// Reset the canvas
function resetCanvas() {
  X = [];
  Y = [];

  resetFunction();
}

function resetFunction() {
  if (a) a.dispose();
  if (b) b.dispose();
  if (c) c.dispose();
  if (d) d.dispose();

  a = tf.tidy(() => tf.scalar(Math.random(1)).variable());
  b = tf.tidy(() => tf.scalar(Math.random(1)).variable());
  c = tf.tidy(() => tf.scalar(Math.random(1)).variable());
  d = tf.tidy(() => tf.scalar(Math.random(1)).variable());
}

// Add point
function addPoint() {
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

function drawCurve() {
  const tensorPlotY = tf.tidy(() => predict(plotX));
  const plotY = tensorPlotY.dataSync();
  tensorPlotY.dispose();

  beginShape();
  noFill();
  stroke(255);
  strokeWeight(2);
  for (let i = 0; i < plotX.length; i++) {
    const x = map(plotX[i], 0, 1, 0, windowWidth);
    const y = map(plotY[i], 0, 1, 0, windowHeight);
    vertex(x, y);
  }
  endShape();
}

// Utils

// Reset button
const button = document.getElementById('reset');
button.addEventListener('click', resetCanvas);

// Radio button
const options = document.getElementsByName('fitting-line');
options.forEach((option) =>
  option.addEventListener('click', () => {
    resetFunction();
    predict = functions[option.value];
  })
);
