import React, { useEffect, useState } from "react";
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  AppBar,
  Toolbar,
  IconButton,
  Drawer,
  List,
  ListItem,
  ListItemText,
  Skeleton,
  CircularProgress,
  useTheme
} from "@mui/material";
import { Bar, Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  PointElement,
  LineElement,
  Filler
} from "chart.js";
import { useNavigate } from "react-router-dom";
import MenuIcon from "@mui/icons-material/Menu";
import TrafficVisualizations from "./components/TrafficVisualizations";
import ModelMetrics from "./components/ModelMetrics";
import AllMetricsPlot from "./components/AllMetrics";

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

const LoadingOverlay = ({ children }) => (
  <Box
    sx={{
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      justifyContent: "center",
      height: "100%",
      width: "100%",
      position: "absolute",
      backgroundColor: "rgba(255, 255, 255, 0.8)",
      zIndex: 1000,
    }}
  >
    {children}
  </Box>
);

const MetricsCard = ({ title, value, color, loading }) => (
  <Card
    sx={{
      height: "100%",
      transition: "transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out",
      "&:hover": {
        transform: "translateY(-5px)",
        boxShadow: 8,
      },
    }}
  >
    <CardContent>
      <Typography variant="h6" gutterBottom sx={{ color: "#666" }}>
        {title}
      </Typography>
      {loading ? (
        <Skeleton variant="rectangular" height={60} />
      ) : (
        <Typography variant="h4" sx={{ color: color }}>
          {value}%
        </Typography>
      )}
    </CardContent>
  </Card>
);

function MetricsPage() {
  const [metrics, setMetrics] = useState({
    accuracy: 0,
    precision: 0,
    recall: 0,
    f1Score: 0,
  });
  const [loading, setLoading] = useState(true);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [userData, setUserData] = useState();
  const navigate = useNavigate();
  const theme = useTheme();

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: true,
        position: "top",
        labels: {
          font: {
            size: 14,
            family: "'Roboto', 'Helvetica', 'Arial', sans-serif",
          },
          padding: 20,
        },
      },
      tooltip: {
        backgroundColor: "rgba(0, 0, 0, 0.8)",
        titleFont: {
          size: 14,
        },
        bodyFont: {
          size: 13,
        },
        padding: 12,
        callbacks: {
          label: (tooltipItem) => `${tooltipItem.raw.toFixed(2)}%`,
        },
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 100,
        grid: {
          color: "rgba(0, 0, 0, 0.1)",
        },
        ticks: {
          font: {
            size: 12,
          },
        },
      },
      x: {
        grid: {
          display: false,
        },
        ticks: {
          font: {
            size: 12,
          },
        },
      },
    },
    animation: {
      duration: 1500,
      easing: "easeInOutQuart",
    },
  };

  const fetchMetrics = async () => {
    setLoading(true);
    try {
      const response = await fetch("http://localhost:5000/get_model_metrics");
      const data = await response.json();

      if (data.status === "success") {
        const { Accuracy, Precision, Recall, "F1 Score": F1Score } = data.metrics.svm_metrics;
        setMetrics({
          accuracy: Accuracy,
          precision: Precision,
          recall: Recall,
          f1Score: F1Score,
        });
      } else {
        console.error("Failed to fetch metrics:", data);
      }
    } catch (error) {
      console.error("Error fetching metrics:", error);
    } finally {
      setTimeout(() => setLoading(false), 800);
    }
  };

  const getUserData = () => {
    const userData = localStorage.getItem("user");
    return userData ? JSON.parse(userData) : null;
  };

  useEffect(() => {
    fetchMetrics();
    setUserData(getUserData());
  }, []);

  const chartData = {
    labels: ["Accuracy", "Precision", "Recall", "F1 Score"],
    datasets: [
      {
        label: "SVM Model Metrics",
        data: [
          metrics.accuracy * 100,
          metrics.precision * 100,
          metrics.recall * 100,
          metrics.f1Score * 100,
        ],
        backgroundColor: [
          "rgba(70, 54, 255, 0.8)",
          "rgba(105, 65, 226, 0.8)",
          "rgba(239, 80, 38, 0.8)",
          "rgba(54, 207, 201, 0.8)",
        ],
        borderColor: [
          "rgb(70, 54, 255)",
          "rgb(105, 65, 226)",
          "rgb(239, 80, 38)",
          "rgb(54, 207, 201)",
        ],
        borderWidth: 2,
        borderRadius: 6,
        hoverBackgroundColor: [
          "rgba(70, 54, 255, 1)",
          "rgba(105, 65, 226, 1)",
          "rgba(239, 80, 38, 1)",
          "rgba(54, 207, 201, 1)",
        ],
      },
    ],
  };

  const trendData = {
    labels: ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
    datasets: [
      {
        label: "Model Performance Trend",
        data: [85, 87, 88, 89, 91, 92],
        borderColor: "rgba(75, 192, 192, 1)",
        backgroundColor: "rgba(75, 192, 192, 0.2)",
        fill: true,
        tension: 0.4,
      },
    ],
  };

  return (
    <>
      <AppBar position="static" sx={{ backgroundColor: "#334455" }}>
        <Toolbar>
          <IconButton
            edge="start"
            color="inherit"
            aria-label="menu"
            onClick={() => setDrawerOpen(true)}
          >
            <MenuIcon />
          </IconButton>
          <Typography variant="h6" sx={{ flexGrow: 1 }}>
            Traffix Route Planner
          </Typography>
        </Toolbar>
      </AppBar>

      <Drawer
        anchor="left"
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
      >
        <Box sx={{ width: 250, p: 2, bgcolor: "#f5f5f5" }}>
          <Typography
            variant="h6"
            sx={{
              color: "#5b5b5b",
              fontWeight: "bold",
              mb: 2,
              textAlign: "center",
              borderBottom: "1px solid #7f8c8d",
              pb: 1,
            }}
          >
            TraffiX
          </Typography>
          <List>
            <ListItem button onClick={() => navigate("/dashboard")}>
              <ListItemText primary="Dashboard" />
            </ListItem>
            <ListItem button onClick={() => navigate("/route-plan")}>
              <ListItemText primary="Route Plan" />
            </ListItem>
            {userData?.userType === "Admin" && (
              <>
                <ListItem button onClick={() => navigate("/block-map")}>
                  <ListItemText primary="Block Routes" />
                </ListItem>
                <ListItem button onClick={() => navigate("/metrics")}>
                  <ListItemText primary="Metrics" />
                </ListItem>
              </>
            )}
          </List>
        </Box>
      </Drawer>

      <Box sx={{ padding: 3, position: "relative" }}>
        {loading && (
          <LoadingOverlay>
            <CircularProgress size={60} />
            <Typography variant="h6" sx={{ mt: 2 }}>
              Loading Metrics...
            </Typography>
          </LoadingOverlay>
        )}

        <Typography variant="h4" gutterBottom sx={{ mb: 4 }}>
          Model Metrics Dashboard
        </Typography>

        <Grid container spacing={4}>
          <Grid item xs={12} md={3}>
            <MetricsCard
              title="Accuracy"
              value={(metrics.accuracy * 100).toFixed(2)}
              color="#4636FF"
              loading={loading}
            />
          </Grid>
          <Grid item xs={12} md={3}>
            <MetricsCard
              title="Precision"
              value={(metrics.precision * 100).toFixed(2)}
              color="#6941E2"
              loading={loading}
            />
          </Grid>
          <Grid item xs={12} md={3}>
            <MetricsCard
              title="Recall"
              value={(metrics.recall * 100).toFixed(2)}
              color="#EF5026"
              loading={loading}
            />
          </Grid>
          <Grid item xs={12} md={3}>
            <MetricsCard
              title="F1 Score"
              value={(metrics.f1Score * 100).toFixed(2)}
              color="#36CFC9"
              loading={loading}
            />
          </Grid>

          <Grid item xs={12}>
            <Card
              sx={{
                height: "100%",
                transition: "transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out",
                "&:hover": {
                  transform: "translateY(-5px)",
                  boxShadow: 8,
                },
              }}
            >
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Metrics Comparison
                </Typography>
                <Box sx={{ height: 400, position: "relative" }}>
                  {loading ? (
                    <Skeleton variant="rectangular" height={400} />
                  ) : (
                    <Line data={chartData} options={chartOptions} />
                  )}
                </Box>
              </CardContent>
            </Card>
          </Grid>
        </Grid>

        <Box sx={{ mt: 4 }}>
          <TrafficVisualizations />
        </Box>
        <Box sx={{ mt: 4 }}>
          <ModelMetrics/>
        </Box>
        <Box sx={{ mt: 4 }}>
          <AllMetricsPlot/>
        </Box>
      </Box>
    </>
  );
}

export default MetricsPage;