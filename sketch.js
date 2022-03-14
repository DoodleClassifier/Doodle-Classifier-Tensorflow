const len = 784;
const total_data = 1000;
const canvas_width = 400;
const canvas_height = 400;



const objects = new Map([
  [0, "Broom"],
  [1, "Door"],
  [2, "Mailbox"],
  [3, "Pencil"],
  [4, "Rainbow"],
  [5, "Sailboat"],
  [6, "Star"]
]);

// Stores the binary files' information of the doodles
let object_data = [];

// Stores training and testing data split objects: [{training: , testing: }, {training: , testing: }]
let object_sets = [];

// Our neural network object
let nn;

// Helper function to load the data from the binary files
function preload() {
  for (let key of objects.keys()) {
    object_data.push(loadBytes('data/' + objects.get(key) + ".bin"));
  }
}

// Splits data into training and testing sets
function prepareData(category, data, label) {
  category.training = [];
  category.testing = [];

  // Split up training and testing data
  for (let i = 0; i < total_data; i++) {
    let offset = i * len;
    let threshold = floor(0.8 * total_data);
    if (i < threshold) {
      category.training[i] = data.bytes.subarray(offset, offset + len);
      category.training[i].label = label;
    } else {
      category.testing[i - threshold] = data.bytes.subarray(offset, offset + len);
      category.testing[i - threshold].label = label;
    }
  }
}

function trainEpoch(training) {
  // Randomizing training data
  // True parameter alters the original array. False or default would create a shuffled copy of the array
  shuffle(training, true);

  // Train the nn for one epoch (one total run through the training data)
  for (let i = 0; i < training.length; i++) {
    // Normanize values by mapping them to range of 0-1
    let data = training[i];
    let inputs = Array.from(data).map(x => x / 255);

    let label = training[i].label;

    let targets = new Array(objects.size).fill(0);
    targets[label] = 1;

    nn.train(inputs, targets);
  }
}

function testAll(testing) {

  let correct = 0;

  for (let i = 0; i < testing.length; i++) {
    // Normanize values by mapping them to range of 0-1
    let data = testing[i];
    let inputs = Array.from(data).map(x => x / 255);
    let label = testing[i].label;
    let guess = nn.predict(inputs);

    // Get index of guess by finding max of output array
    let m = max(guess)
    let classification = guess.indexOf(max(m));

    if (classification === label) {
      correct++;
    }
  }
  let percent = 100 * correct / testing.length;
  return percent;
}

function formatDecimal(num) {
  return nf(num, 2, 2);
}

// Helper function to initialize guess text box on page load and when clearing image
function zeroGuess(){
  let guessDiv = select("#guessDiv");
  let zeroGuess = "";

  let guessText = makeGuessStrings(new Array(objects.size).fill(0));

  for (let str of guessText){
    zeroGuess += str.string;
  }

  guessDiv.html(zeroGuess);

}

// Helper function to generate an array of strings, one for each object and the predicted percentage by the model. Takes in guess which is simply an array of the predicted percentages.
function makeGuessStrings(guess){
  guessText = [];
  
  // Generate strings and associate them with the guess values
  for (let key of objects.keys()) {
    guessText.push({});
    guessText[key].string = objects.get(key) + "<span style='float:right;'>" + formatDecimal(guess[key] * 100) + "%</span><br>";
    guessText[key].value = guess[key];
  }

  // Sort guess text array by descending value
  guessText.sort(function(a, b) {
    if (a.value < b.value){
      return 1;
    } else if (a.value > b.value) {
      return -1;
    }
    return 0;
  });

  return guessText;
}

function guess() {
  let inputs = [];
  let img = get();
  img.resize(28, 28);
  img.loadPixels();

  // Turn pixels into byte array
  for (let i = 0; i < len; i++) {
    let bright = img.pixels[i * 4];
    inputs[i] = bright / 255.0;
  }

  let guess = nn.predict(inputs);
  let m = max(guess);
  // let classification = guess.indexOf(m);

  let guessDiv = select("#guessDiv");

  // Format text
  let guessText = makeGuessStrings(guess);

  // Color the top result black with a span, while other results will be grey
  guessText[0].string = "<span style='color:black'>" + guessText[0].string + "</span>";

  // Create string from guess text
  let guessString = "";

  for (let str of guessText){
    guessString += str.string;
  }

  guessDiv.html(guessString);
}

// Setup function of the p5.js library, basically it runs this function on page load
function setup() {
  let canvas = createCanvas(canvas_width, canvas_height);
  canvas.background(0);
  canvas.parent("canvasDiv")
  zeroGuess();

  for (let key of objects.keys()) {
    object_sets.push({});
    prepareData(object_sets[key], object_data[key], key);
  }

  // Make nn object, 784 inputs (one per pixel value), 64 nodes in hidden layer, x outputs for all objects
  
  nn = new NeuralNetwork(784, 60, objects.size);
    // nn = RandomForestClassifier(200);

  // Preparing the data
  let training = [];
  for (let key of objects.keys()) {
    training = training.concat(object_sets[key].training);
  }

  let testing = [];
  for (let key of objects.keys()) {
    testing = testing.concat(object_sets[key].testing);
  }

  let trainButton = select('#train');
  let epochCounter = 0;
  trainButton.mousePressed(function () {
    trainEpoch(testing);
    epochCounter++;
    console.log("Epoch: " + epochCounter);
  });

  let testButton = select('#test');
  testButton.mousePressed(function () {
    let percent = testAll(testing);
    console.log("Percent: " + formatDecimal(percent) + "%");
  });
  
  // Logic for "guess when drawing on canvas"
  let mouseDown = false;
  canvas.mousePressed(function(){
    mouseDown = true;
  });

  canvas.mouseReleased(function(){
    mouseDown = false;
  });

  canvas.mouseMoved(function(){
    if (mouseDown){
      guess();
    }
  });

  let clearButton = select('#clear');
  clearButton.mousePressed(function () {
    background(0);
    zeroGuess();
  });
}

// Draw function of p5.js library, this function is run each frame to check for updates
function draw() {
  strokeWeight(16);
  stroke(255);
  if (mouseIsPressed) {
    line(pmouseX, pmouseY, mouseX, mouseY);
  }
}


/*
Notes for improvements:
- convolutional layer before final layer
- softmax 
- cross entropy

*/


