// src/App.js
import React, { useEffect, useState, useRef } from "react";
import * as tf from "@tensorflow/tfjs";

/**
 * Forecast — Inventory Reorder Predictor (standalone)
 * - Generates mock products in-browser (no API)
 * - Creates a deterministic 'reorder' label from a business rule
 * - Trains a TF.js model in-browser and shows predictions
 */

function generateProducts(count = 150) {
  const names = [
    "Choco Bar","Soda Pack","Rice 5kg","Coffee Beans","Toothpaste","Shampoo","Notebook","Pen",
    "Soap Bar","Cereal","Juice Bottle","Water Bottle","Ketchup","Mayonnaise","Cookies","Tea Box",
    "Nuts Pack","Olive Oil","Pasta","Sauce"
  ];
  const products = [];
  for (let i = 1; i <= count; i++) {
    const name = `${names[i % names.length]} ${i}`;
    // varied data
    const avgSalesPerWeek = Math.max(1, Math.round(Math.abs(Math.sin(i * 11) * 120)));
    const currentInventory = Math.max(0, Math.round(Math.abs(Math.cos(i * 7) * 400)));
    const daysToReplenish = 1 + (i % 21);

    // Business rule: expected consumption during lead time * safety factor
    const expectedDuringLead = avgSalesPerWeek * (daysToReplenish / 7);
    const safetyFactor = 1.25;
    const reorder = currentInventory < expectedDuringLead * safetyFactor ? 1 : 0;

    products.push({
      id: i,
      name,
      currentInventory,
      avgSalesPerWeek,
      daysToReplenish,
      reorder
    });
  }
  return products;
}

function normalize(tensor, min, max) {
  const sub = tf.sub(tensor, min);
  const range = tf.sub(max, min);
  return tf.divNoNan(sub, range);
}

export default function App() {
  const [products, setProducts] = useState([]);
  const [runningModel, setRunningModel] = useState(false);
  const [stats, setStats] = useState({
    total: 0,
    serverReorders: 0,
    modelReorders: 0,
    modelAccuracy: "N/A",
  });
  const modelStore = useRef(null);

  useEffect(() => {
    // generate mock products on load
    const data = generateProducts(150);
    setProducts(data);
    setStats((s) => ({ ...s, total: data.length, serverReorders: data.reduce((acc, p) => acc + (p.reorder ? 1 : 0), 0) }));

    return () => {
      if (modelStore.current && modelStore.current.model) {
        try {
          modelStore.current.model.dispose();
        } catch (e) {}
      }
    };
  }, []);

  async function buildAndTrainModel(items) {
    const xs = items.map((p) => [p.currentInventory, p.avgSalesPerWeek, p.daysToReplenish]);
    const ys = items.map((p) => [p.reorder]);

    const xTensor = tf.tensor2d(xs);
    const yTensor = tf.tensor2d(ys);

    const xMin = xTensor.min(0);
    const xMax = xTensor.max(0);
    const xNorm = normalize(xTensor, xMin, xMax);

    const model = tf.sequential();
    model.add(tf.layers.dense({ inputShape: [3], units: 24, activation: "relu" }));
    model.add(tf.layers.dense({ units: 12, activation: "relu" }));
    model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));

    model.compile({
      optimizer: tf.train.adam(0.01),
      loss: "binaryCrossentropy",
      metrics: ["accuracy"],
    });

    const total = items.length;
    const split = Math.floor(total * 0.8);
    const xTrain = xNorm.slice([0, 0], [split, -1]);
    const yTrain = yTensor.slice([0, 0], [split, -1]);
    const xVal = xNorm.slice([split, 0], [total - split, -1]);
    const yVal = yTensor.slice([split, 0], [total - split, -1]);

    const history = await model.fit(xTrain, yTrain, {
      epochs: 80,
      batchSize: 16,
      shuffle: true,
      validationData: [xVal, yVal],
      verbose: 0,
    });

    const lastValAcc =
      history.history.val_accuracy && history.history.val_accuracy.length
        ? history.history.val_accuracy[history.history.val_accuracy.length - 1]
        : null;

    if (modelStore.current && modelStore.current.model) {
      try {
        modelStore.current.model.dispose();
      } catch (e) {}
    }
    modelStore.current = { model, xMin, xMax };

    // dispose intermediate tensors
    xTensor.dispose();
    yTensor.dispose();
    xNorm.dispose();
    xTrain.dispose();
    yTrain.dispose();
    xVal.dispose();
    yVal.dispose();

    return { model, xMin, xMax, valAcc: lastValAcc };
  }

  async function trainAndPredict() {
    if (!products || products.length === 0) {
      alert("No products available.");
      return;
    }
    setRunningModel(true);
    try {
      const { model, xMin, xMax, valAcc } = await buildAndTrainModel(products);

      const inputs = products.map((p) => [p.currentInventory, p.avgSalesPerWeek, p.daysToReplenish]);
      const inputTensor = tf.tensor2d(inputs);
      const inputNorm = normalize(inputTensor, xMin, xMax);

      const preds = model.predict(inputNorm);
      const predData = await preds.data();

      const updated = products.map((p, i) => {
        const score = predData[i];
        const predicted = score > 0.5 ? 1 : 0;
        return { ...p, predictionScore: score, prediction: predicted, predictionText: predicted ? "Reorder" : "No Reorder" };
      });

      const modelReorders = updated.reduce((s, x) => s + (x.prediction ? 1 : 0), 0);
      setProducts(updated);
      setStats({
        total: updated.length,
        serverReorders: updated.reduce((s, p) => s + (p.reorder ? 1 : 0), 0),
        modelReorders,
        modelAccuracy: valAcc != null ? (valAcc * 100).toFixed(1) + "%" : "N/A",
      });

      inputTensor.dispose();
      inputNorm.dispose();
      preds.dispose();
    } catch (err) {
      console.error(err);
      alert("Error training/predicting: " + (err && err.message ? err.message : err));
    } finally {
      setRunningModel(false);
    }
  }

  function downloadCSV() {
    if (!products || products.length === 0) {
      alert("No products to download.");
      return;
    }
    const header = ["id", "name", "currentInventory", "avgSalesPerWeek", "daysToReplenish", "serverReorder", "prediction", "predictionScore"];
    const rows = products.map((p) =>
      [
        p.id,
        `"${String(p.name).replace(/"/g, '""')}"`,
        p.currentInventory,
        p.avgSalesPerWeek,
        p.daysToReplenish,
        p.reorder,
        p.prediction ?? "",
        p.predictionScore ? p.predictionScore.toFixed(4) : "",
      ].join(",")
    );
    const csv = [header.join(","), ...rows].join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "products_with_predictions.csv";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  // Inline styles
  const containerStyle = { padding: 20, fontFamily: "Inter, Roboto, Arial, sans-serif", background: "#f5f7fb", minHeight: "100vh" };
  const cardStyle = { background: "#fff", padding: 12, borderRadius: 8, boxShadow: "0 2px 6px rgba(0,0,0,0.06)" };
  const th = { textAlign: "left", padding: 8, borderBottom: "1px solid #eee" };
  const td = { padding: 8, borderBottom: "1px solid #f2f4f8" };

  return (
    <div style={containerStyle}>
      <div style={{ maxWidth: 1100, margin: "0 auto" }}>
        <h1 style={{ margin: 0 }}>Forecast — Inventory Reorder Predictor</h1>
        <p style={{ color: "#555" }}>Standalone demo: mock data generated in-browser, TF.js training and predictions client-side.</p>

        <div style={{ display: "flex", gap: 12, marginBottom: 12 }}>
          <div style={{ ...cardStyle, flex: "1 1 0" }}>
            <div style={{ fontSize: 12, color: "#666" }}>Products</div>
            <div style={{ fontSize: 20, fontWeight: 700 }}>{stats.total}</div>
          </div>
          <div style={{ ...cardStyle, flex: "1 1 0" }}>
            <div style={{ fontSize: 12, color: "#666" }}>Server Reorders (rule)</div>
            <div style={{ fontSize: 20, fontWeight: 700 }}>{stats.serverReorders}</div>
          </div>
          <div style={{ ...cardStyle, flex: "1 1 0" }}>
            <div style={{ fontSize: 12, color: "#666" }}>Model Predicted Reorders</div>
            <div style={{ fontSize: 20, fontWeight: 700 }}>{stats.modelReorders}</div>
          </div>
          <div style={{ ...cardStyle, flex: "1 1 0" }}>
            <div style={{ fontSize: 12, color: "#666" }}>Model Val Accuracy</div>
            <div style={{ fontSize: 20, fontWeight: 700 }}>{stats.modelAccuracy}</div>
          </div>
        </div>

        <div style={{ display: "flex", gap: 8, marginBottom: 12 }}>
          <button onClick={() => { const data = generateProducts(150); setProducts(data); setStats((s) => ({ ...s, total: data.length, serverReorders: data.reduce((a, b) => a + (b.reorder ? 1 : 0), 0) })); }}>
            Regenerate Products
          </button>
          <button onClick={trainAndPredict} disabled={runningModel || products.length === 0}>
            {runningModel ? "Training..." : "Train Model & Predict"}
          </button>
          <button onClick={downloadCSV} style={{ background: "transparent", color: "#2563eb", border: "1px solid #2563eb" }}>
            Download CSV
          </button>
        </div>

        <div style={{ ...cardStyle }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
            <h3 style={{ margin: 0 }}>Products Table</h3>
            <div style={{ fontSize: 12, color: "#666" }}>{products.length} items</div>
          </div>

          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead>
                <tr>
                  <th style={th}>ID</th>
                  <th style={th}>Name</th>
                  <th style={th}>Inventory</th>
                  <th style={th}>Avg sales / wk</th>
                  <th style={th}>Days to replenish</th>
                  <th style={th}>Rule label</th>
                  <th style={th}>Prediction</th>
                  <th style={th}>Score</th>
                </tr>
              </thead>
              <tbody>
                {products.map((p) => (
                  <tr key={p.id}>
                    <td style={td}>{p.id}</td>
                    <td style={td}>{p.name}</td>
                    <td style={td}>{p.currentInventory}</td>
                    <td style={td}>{p.avgSalesPerWeek}</td>
                    <td style={td}>{p.daysToReplenish}</td>
                    <td style={td}>{p.reorder}</td>
                    <td style={td}>{p.predictionText ?? "-"}</td>
                    <td style={td}>{p.predictionScore != null ? Number(p.predictionScore).toFixed(3) : "-"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div style={{ marginTop: 12, color: "#666", fontSize: 13 }}>
          <div>Notes:</div>
          <ul>
            <li>This app is self-contained and requires only your React dev server at <code>http://localhost:3000</code>.</li>
            <li>Training runs in the browser — it may take several seconds depending on dataset size and CPU.</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
