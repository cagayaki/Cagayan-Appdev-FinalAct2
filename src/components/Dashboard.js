import React from "react";

export default function Dashboard({ stats }) {
  return (
    <div style={{ display: "flex", gap: 12, marginBottom: 12 }}>
      <StatCard title="Products" value={stats.total} />
      <StatCard title="Server Reorders" value={stats.serverReorders} />
      <StatCard title="Model Predicted Reorders" value={stats.modelReorders} />
      <StatCard title="Model Accuracy (val)" value={stats.modelAccuracy ?? "—"} />
    </div>
  );
}

function StatCard({ title, value }) {
  return (
    <div style={{ flex: 1, padding: 12, borderRadius: 8, boxShadow: "0 2px 6px rgba(0,0,0,0.06)", background: "#fff" }}>
      <div style={{ fontSize: 12, color: "#666" }}>{title}</div>
      <div style={{ fontSize: 22, fontWeight: 700 }}>{value}</div>
    </div>
  );
}
