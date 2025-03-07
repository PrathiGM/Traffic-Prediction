import { useState, useEffect } from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import AuthPage from "./Login/AuthPage";
import Dashboard from "./Dashboard/Dashboard";
import RoutePlanner from "./Predictroute";
import MapWithAdminControls from "./Dashboard/BlockedRoutes";
import MetricsPage from "./ShowAccuracy";

function App() {
  const [count, setCount] = useState(0);

  return (
    <BrowserRouter>
      <Routes>
        <Route index path="/" element={<AuthPage />} />
        <Route index path="/dashboard" element={<Dashboard />} />
        <Route index path="/route-plan" element={<RoutePlanner />} />
        <Route index path="/block-map" element={<MapWithAdminControls />} />
        <Route index path="/metrics" element={<MetricsPage />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
