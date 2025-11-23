import React, { useState } from "react";
import * as tf from "@tensorflow/tfjs";
export default function InventoryPredictor() {
const [prediction, setPrediction] = useState(null);
// Example training data (stock, avgSales, leadTime)
const trainingData = tf.tensor2d([
[20, 50, 3],
[5, 30, 5],
[15, 40, 4],
[8, 60, 2],
]); // creates a 2D tensor (matrix) from the provided data.
// Labels: 1 = reorder, 0 = don't reorder
const outputData = tf.tensor2d([[0], [1], [0], [1]]);
const handlePredict = async () => {
// 1. Create model
const model = tf.sequential();
model.add(
tf.layers.dense({ inputShape: [3], units: 8, activation: "relu"
})
);
model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));

// 2. Compile model
model.compile({
optimizer: "adam",
loss: "binaryCrossentropy",
metrics: ["accuracy"],
});
// 3. Train model
await model.fit(trainingData, outputData, {
epochs: 200,
shuffle: true,
});
// 4. Predict a new product
const newProduct = tf.tensor2d([[10, 45, 3]]);
const result = model.predict(newProduct);
const value = (await result.data())[0];
setPrediction(value > 0.5 ? "Reorder" : "No Reorder");
};
return (
<div style={{ padding: 20 }}>
<h2>Inventory Reorder Predictor</h2>
<button onClick={handlePredict}>Predict</button>
{prediction && <p>Prediction: {prediction}</p>}
</div>
);
}