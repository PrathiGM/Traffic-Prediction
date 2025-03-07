import React, { useEffect, useMemo, useState } from "react";
import { MapContainer, TileLayer, Marker, Popup, useMap } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import "./Dashboard.css";
import "leaflet.heat";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import {
  faTachometerAlt,
  faCar,
  faClock,
  faRoad,
  faExclamationCircle,
  faShieldAlt,
} from "@fortawesome/free-solid-svg-icons";
import axios from "axios";
import {
  AppBar,
  Toolbar,
  Drawer,
  List,
  ListItem,
  ListItemText,
  IconButton,
  Typography,
  Button,
  Card,
  CardContent,
  Grid,
  Box,
  Chip,
} from "@mui/material";
import MenuIcon from "@mui/icons-material/Menu";
import WbSunnyIcon from "@mui/icons-material/WbSunny";
import TrafficIcon from "@mui/icons-material/Traffic";
import LocationOnIcon from "@mui/icons-material/LocationOn";
import ArticleIcon from "@mui/icons-material/Article";
import { styled } from "@mui/system";
import { useNavigate } from "react-router-dom";
import { db } from "../firebase";
import {
  Form,
  Row,
  Col,
  Container,
  Alert,
  Spinner,
  Badge,
} from "react-bootstrap";
import "react-responsive-carousel/lib/styles/carousel.min.css"; // requires a loader
import { Carousel } from "react-responsive-carousel";
import MobileStepper from "@material-ui/core/MobileStepper";
import KeyboardArrowRight from "@material-ui/icons/KeyboardArrowRight";
import KeyboardArrowLeft from "@material-ui/icons/KeyboardArrowLeft";
import ErrorBoundary from "../ErrorBoundary";
import aibot from "../components/aijson.json";
import Lottie from "lottie-react";
import Typist from "react-typist";
import "react-typist/dist/Typist.css";
import TypingEffect from "../components/TextTyping";
import InfoCard from "../components/InfoCard";
import Sentiment from "sentiment";

const MyCollection = [
  {
    label: "First Picture",
    imgPath:
      "https://media.geeksforgeeks.org/wp-content/uploads/20210208000010/1.png",
  },
  {
    label: "Second Picture",
    imgPath:
      "https://media.geeksforgeeks.org/wp-content/uploads/20210208000009/2.png",
  },
  {
    label: "Third Picture",
    imgPath:
      "https://media.geeksforgeeks.org/wp-content/uploads/20210208000008/3.png",
  },
];

const paragraph =
  "       Our Traffic Prediction System leverages historical traffic data to predict future traffic trends and provide optimal routes for better travel planning. By analyzing patterns, it forecasts traffic conditions for the next 7 days, ensuring efficiency, reduced congestion, and enhanced user experience.";
const DrawerContent = styled(Box)({
  width: 250,
  padding: "1rem",
  backgroundColor: "#f5f5f5",
});

const DashboardContainer = styled(Box)({
  minHeight: "100vh",
  backgroundColor: "#eef1f5",
  padding: "2rem",
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
});

const HeroSection = styled(Box)({
  width: "100%",
  padding: "2rem",
  background: "linear-gradient(120deg, #007bff, #007bff)",
  color: "#fff",
  borderRadius: "16px",
  marginBottom: "2rem",
  display: "flex",
  flexDirection: "row",
  justifyContent: "space-between",
  gap: "2rem",
});

const HeroContent = styled(Box)({
  display: "flex",
  flexDirection: "column",
  justifyContent: "center",
});

const StyledCard = styled(Card)({
  backgroundColor: "#fff  ",

  borderRadius: "12px",
  boxShadow: "0px 6px 20px rgba(0, 0, 0, 0.1)",
  padding: "1.5rem",
  transition: "transform 0.3s ease-in-out",
  "&:hover": {
    transform: "scale(1.05)",
  },
  textAlign: "center",
  border: "1px solid orange",
  height: 150,
});

const NewsList = styled(Box)({
  width: "99.5%",
  marginTop: "2rem",
  border: "1px solid orange",
  borderRadius: 16,
  background:
    "linear-gradient(45deg, rgba(106,17,203,0.2),rgba(37,117,252,0.2));",
  height: "100%",
});
const HeatmapLayer = ({ data }) => {
  const map = useMap();

  useEffect(() => {
    if (!data.length) return;

    const heatLayer = L.heatLayer(data, {
      radius: 25,
      blur: 15,
      maxZoom: 17,
    }).addTo(map);

    return () => {
      map.removeLayer(heatLayer);
    };
  }, [data, map]);

  return null;
};
const Dashboard = () => {
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [location, setLocation] = useState(null);
  const [weather, setWeather] = useState(null);
  const [traffic, setTraffic] = useState(null);
  const [news, setNews] = useState([]);
  const [heatmapData, setHeatmapData] = useState([]);
  const [trafficData, setTrafficData] = useState(null);
  const [predictedData, setPredictedData] = useState(null);
  const sentiment = new Sentiment();

  const [selectedRegion, setSelectedRegion] = useState({
    name: "Bangalore",
    latitude: 12.9716,
    longitude: 77.5946,
  });

  const [userData, setuserData] = useState();
  const memoizedText = useMemo(() => paragraph, [paragraph]);
  const [shouldRender, setShouldRender] = useState(false);

  useEffect(() => {
    setShouldRender(true); // Trigger only once when the component mounts.
  }, []);
  const getUserData = () => {
    const userData = localStorage.getItem("user");
    return userData ? JSON.parse(userData) : null;
  };
  const [regions] = useState([
    { name: "Bangalore", latitude: 12.9716, longitude: 77.5946 },
    { name: "Mumbai", latitude: 19.076, longitude: 72.8777 },
    { name: "Delhi", latitude: 28.7041, longitude: 77.1025 },
  ]);

  useEffect(() => {
    setuserData(getUserData());
  }, []);
  const navigate = useNavigate();
  useEffect(() => {
    navigator.geolocation.getCurrentPosition((position) => {
      const { latitude, longitude } = position.coords;
      setLocation({ latitude, longitude });
    });
  }, []);
  function ChangeMapView({ center }) {
    const map = useMap();
    map.setView(center, map.getZoom());
    return null;
  }
  const handleRegionChange = async (e) => {
    const selectedRegion = regions.find(
      (region) => region.name === e.target.value
    );
    if (selectedRegion) {
      const { name, latitude, longitude } = selectedRegion;
      setSelectedRegion(selectedRegion);
      // setFormData({ ...formData, region: name, latitude, longitude });
      setLocation({ latitude, longitude });
    }
  };

  useEffect(() => {
    const fetchWeather = async () => {
      if (!location) return;
      const { data } = await axios.get(
        `https://api.openweathermap.org/data/2.5/weather?lat=${location.latitude}&lon=${location.longitude}&appid=4b3120747b9ef1f222316cf9cf47bd7a&units=metric`
      );
      setWeather(data);
    };

    const fetchTraffic = async () => {
      if (!location) return;
      const { data } = await axios.get(
        `https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?point=${location.latitude},${location.longitude}&key=5QpZLwcTD1Gz8dJ0O7o1u9vGokfRF1Og`
      );
      setTraffic(data);
    };

    const fetchNews = async () => {
      const { data } = await axios.get(
        `https://api.bing.microsoft.com/v7.0/news/search?q=traffic&count=10`,
        {
          headers: {
            "Ocp-Apim-Subscription-Key": "a70199da760347429bf6df19030da50f",
          },
        }
      );
      setNews(
        data.value.map((item) => {
          return {
            ...item,
            sentimentScore: sentiment.analyze(item.description).score,
            sentimentComparative: sentiment.analyze(item.description)
              .comparative,
          };
        })
      );
    };
    const fetchTrafficData = async () => {
      if (!location) return;
      const { latitude, longitude } = location;
      const { data } = await axios.get(
        `https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?point=${latitude},${longitude}&key=5QpZLwcTD1Gz8dJ0O7o1u9vGokfRF1Og`
      );
      console.log(data);
      const processedData = data?.flowSegmentData?.coordinates?.coordinate?.map(
        (coord) => [
          coord.latitude,
          coord.longitude,
          1, // Weight for heatmap intensity (can be adjusted based on traffic speed)
        ]
      );
      setHeatmapData(processedData);
    };
    const generateBBox = (lat, lon, delta = 0.01) => {
      return `${lon - delta},${lat - delta},${lon + delta},${lat + delta}`;
    };

    // Fetch Traffic Stats
    const fetchTrafficStats = async () => {
      if (!location) return;
      const { latitude, longitude } = location;

      try {
        const response = await axios.get(
          `https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json`,
          {
            params: {
              point: `${latitude},${longitude}`,
              key: "5QpZLwcTD1Gz8dJ0O7o1u9vGokfRF1Og",
            },
          }
        );

        const trafficData = response.data.flowSegmentData;
        console.log("Traffic Data:", trafficData);

        setTrafficData({
          currentSpeed: trafficData.currentSpeed,
          freeFlowSpeed: trafficData.freeFlowSpeed,
          currentTravelTime: trafficData.currentTravelTime,
          freeFlowTravelTime: trafficData.freeFlowTravelTime,
          roadClosure: trafficData.roadClosure,
          confidence: trafficData.confidence,
        });
        setShouldRender(true);
      } catch (error) {
        console.error("Error fetching traffic stats:", error);
      }
    };
    fetchWeather();
    fetchTraffic();
    fetchNews();
    fetchTrafficData();
    fetchTrafficStats();
  }, [location]);

  useEffect(() => {
    const fetchPredictions = async () => {
      const trafficResponse = await axios.get(
        `https://api.tomtom.com/traffic/services/4/flowSegmentData/relative0/10/json?point=${selectedRegion?.latitude},${selectedRegion?.longitude}&key=5QpZLwcTD1Gz8dJ0O7o1u9vGokfRF1Og`
      );
      const trafficDataw = trafficResponse.data.flowSegmentData;

      // Generate a `currentTrafficData` array dynamically from the TomTom data
      // Example: [currentSpeed, freeFlowSpeed, confidence] as mock data points
      const currentTrafficData = [
        trafficDataw.currentSpeed,
        trafficDataw.freeFlowSpeed,
        Math.round(trafficDataw.confidence * 100), // Convert confidence to percentage
      ];
      const response = await axios.post("http://localhost:5000/predict_lstm", {
        region: selectedRegion?.name,
        latitude: selectedRegion?.latitude,
        longitude: selectedRegion?.longitude,
        weatherData: JSON.stringify({
          temperature: (weather.main.temp - 273.15).toFixed(2), // Convert Kelvin to Celsius
          condition: weather.weather[0].description,
        }),
        trafficData: JSON.stringify({
          speed: trafficResponse.data.flowSegmentData.currentSpeed,
          freeFlowSpeed: trafficResponse.data.flowSegmentData.freeFlowSpeed,
          congestionLevel: trafficResponse.data.flowSegmentData.confidence,
        }),
        news: news.map((item) => item.name),
        currentTrafficData: currentTrafficData,
      });

      const trafficData = trafficResponse.data.flowSegmentData;

      // Calculate the new volumes based on traffic data
      console.log(response.data);
      // const adjustedPredictions = Object.entries(
      //   response.data.predictions
      // ).reduce((acc, [interval, prediction]) => {
      //   // Assuming trafficData.speed and freeFlowSpeed are available for calculation
      //   const adjustmentFactor =
      //     trafficData.currentSpeed / trafficData.freeFlowSpeed;
      //   const updatedVolume = prediction.volume * adjustmentFactor;

      //   acc[interval] = {
      //     ...prediction,
      //     volume: updatedVolume, // Adjusted volume based on the current traffic data
      //   };
      //   return acc;
      // });
      const adjustedPredictions = Object.entries(
        response.data.predictions
      ).reduce((acc, [interval, prediction]) => {
        // Mock adjustment factor
        const adjustmentFactor =
          trafficData.currentSpeed / trafficData.freeFlowSpeed;

        // Adjust the volume for more realism
        const timeFactor = interval.toLowerCase().includes("hour")
          ? 1
          : interval.toLowerCase().includes("day")
          ? 1.5
          : interval.toLowerCase().includes("week")
          ? 2
          : interval.toLowerCase().includes("month")
          ? 2.5
          : interval.toLowerCase().includes("year")
          ? 3
          : 1; // Default value if no match
        const updatedVolume = Math.fround(
          prediction.volume * adjustmentFactor * timeFactor
        ).toFixed(2);

        acc[interval] = {
          ...prediction,
          volume: updatedVolume, // Adjusted volume based on speed and interval
        };

        // Dynamically adjust category based on volume
        if (updatedVolume > 120) acc[interval].category = "High";
        else if (updatedVolume > 60) acc[interval].category = "Medium";
        else acc[interval].category = "Low";

        return acc;
      }, {});
      console.log(adjustedPredictions);
      setPredictedData(adjustedPredictions);
    };
    if (selectedRegion && weather) {
      fetchPredictions();
    }
  }, [selectedRegion, weather]);
  const TrafficStatsCard = ({ trafficData }) => {
    if (!trafficData) return <p>Loading traffic data...</p>;

    const stats = [
      {
        icon: faTachometerAlt,
        label: "Current Speed",
        value: `${trafficData.currentSpeed} km/h`,
      },
      {
        icon: faCar,
        label: "Free Flow Speed",
        value: `${trafficData.freeFlowSpeed} km/h`,
      },
      {
        icon: faClock,
        label: "Travel Time (Current)",
        value: `${(trafficData.currentTravelTime / 60).toFixed(2)} mins`,
      },
      {
        icon: faClock,
        label: "Travel Time (Free Flow)",
        value: `${(trafficData.freeFlowTravelTime / 60).toFixed(2)} mins`,
      },
      {
        icon: faRoad,
        label: "Road Closure",
        value: trafficData.roadClosure ? "Yes" : "No",
      },
      {
        icon: faShieldAlt,
        label: "Confidence Score",
        value: trafficData.confidence,
      },
    ];

    return (
      <Card
        className="traffic-grid"
        elevation={5}
        sx={{
          padding: "20px",
          borderRadius: "10px",
          boxShadow: "0 4px 8px rgba(0, 0, 0, 0.1)",
          width: "96.5%",
          margin: "auto",
          marginTop: "15px",
          border: "1px solid orange",
          backgroundImage:
            "linear-gradient(45deg, rgba(106,17,203,0.2),rgba(37,117,252,0.2))",
        }}
      >
        <Typography
          variant="h4"
          textAlign="center"
          sx={{ mb: 3 }}
          fontWeight="400"
        >
          Traffic Statistics
        </Typography>
        <div className="grid-container">
          {stats.map((stat, index) => (
            <div className="grid-item" key={index}>
              <FontAwesomeIcon icon={stat.icon} className="grid-icon" />
              <div>
                <h4>{stat.label}</h4>
                <p>{stat.value}</p>
              </div>
            </div>
          ))}
        </div>
      </Card>
    );
  };
  const getBadgeVariant = (category) => {
    switch (category) {
      case "Low":
        return "success";
      case "Medium":
        return "warning";
      case "High":
        return "danger";
      default:
        return "secondary";
    }
  };

  return (
    <ErrorBoundary>
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
              Dashboard
            </Typography>
          </Toolbar>
        </AppBar>

        <Drawer
          anchor="left"
          open={drawerOpen}
          onClose={() => setDrawerOpen(false)}
          style={{
            width: "300px",
            backgroundColor: "#2c3e50", // Dark background for a modern look
            color: "#ecf0f1", // Light text for contrast
            padding: "10px",
          }}
        >
          <DrawerContent
            style={{
              padding: "20px",
              display: "flex",
              flexDirection: "column",
              height: "100%",
            }}
          >
            <Typography
              variant="h6"
              style={{
                color: "#5b5b5b",
                fontWeight: "bold",
                marginBottom: "20px",
                textAlign: "center",
                borderBottom: "1px solid #7f8c8d",
                paddingBottom: "10px",
              }}
            >
              TraffiX
            </Typography>
            <List style={{ flexGrow: 1 }}>
              <ListItem button style={{ marginBottom: "10px" }}>
                <ListItemText
                  primary="Dashboard"
                  primaryTypographyProps={{
                    style: {
                      fontSize: "16px",
                      fontWeight: "500",
                      color: "#5b5b5b",
                    },
                  }}
                  onClick={() => {
                    navigate("/dashboard");
                  }}
                />
              </ListItem>
              <ListItem button style={{ marginBottom: "10px" }}>
                <ListItemText
                  primary="Route Plan"
                  primaryTypographyProps={{
                    style: {
                      fontSize: "16px",
                      fontWeight: "500",
                      color: "#5b5b5b",
                    },
                  }}
                  onClick={() => {
                    navigate("/route-plan");
                  }}
                />
              </ListItem>
              {userData?.userType === "Admin" && (
                <>
                  <ListItem button style={{ marginBottom: "10px" }}>
                    <ListItemText
                      primary="Block Routes"
                      primaryTypographyProps={{
                        style: {
                          fontSize: "16px",
                          fontWeight: "500",
                          color: "#5b5b5b",
                        },
                      }}
                      onClick={() => {
                        navigate("/block-map");
                      }}
                    />
                  </ListItem>
                  <ListItem button style={{ marginBottom: "10px" }}>
                    <ListItemText
                      primary="Metrics"
                      primaryTypographyProps={{
                        style: {
                          fontSize: "16px",
                          fontWeight: "500",
                          color: "#5b5b5b",
                        },
                      }}
                      onClick={() => {
                        navigate("/metrics");
                      }}
                    />
                  </ListItem>
                </>
              )}
            </List>
            <div
              style={{
                textAlign: "center",
                borderTop: "1px solid #7f8c8d",
                padding: "10px 0",
                marginTop: "20px",
              }}
            >
              <Typography
                variant="caption"
                style={{ color: "#bdc3c7", fontSize: "12px" }}
              >
                © 2024 Traffic Insights
              </Typography>
            </div>
          </DrawerContent>
        </Drawer>

        <DashboardContainer>
          <div
            style={{
              width: "100%",
              maxHeight: "500px",
              overflow: "hidden",
              marginBottom: 20,
              background: "linear-gradient(45deg, #6a11cb, #2575fc)",
            }}
          >
            <Carousel
              showThumbs={false}
              showArrows={true}
              autoPlay
              infiniteLoop
              dynamicHeight={false}
              showStatus={false}
              interval={3000}
            >
              <div>
                <img
                  src="https://www.shutterstock.com/shutterstock/photos/1815836900/display_1500/stock-photo-san-diego-california-usa-jan-emergency-auto-on-busy-intercity-freeway-paramedic-1815836900.jpg"
                  alt="Slide 1"
                  style={{ objectFit: "cover", height: "500px", width: "100%" }}
                />
              </div>
              <div>
                <img
                  src="https://www.shutterstock.com/shutterstock/photos/2448609629/display_1500/stock-photo--new-york-usa-traffic-jam-on-the-brooklyn-queens-expressway-interstate-it-was-2448609629.jpg"
                  alt="Slide 2"
                  style={{ objectFit: "cover", height: "500px", width: "100%" }}
                />
              </div>
              <div>
                <img
                  src="https://www.shutterstock.com/shutterstock/photos/2448580065/display_1500/stock-photo--new-york-usa-traffic-jam-on-the-brooklyn-queens-expressway-interstate-it-was-2448580065.jpg"
                  alt="Slide 3"
                  style={{ objectFit: "cover", height: "500px", width: "100%" }}
                />
              </div>
            </Carousel>
          </div>

          <InfoCard userData={userData} />
          <Grid container spacing={3}>
            <Grid item xs={12} sm={4}>
              <StyledCard
                style={{ backgroundColor: "rgba(242, 121, 53, 0.7)" }}
              >
                <WbSunnyIcon fontSize="large" sx={{ color: "#EC5800" }} />
                <Typography variant="h6">Weather</Typography>
                <Typography>
                  {weather
                    ? `${weather.main.temp}°C - ${weather.weather[0].description}`
                    : "Loading..."}
                </Typography>
              </StyledCard>
            </Grid>
            <Grid item xs={12} sm={4}>
              <StyledCard
                style={{ backgroundColor: "rgba(135, 211, 124, 0.7" }}
              >
                <TrafficIcon fontSize="large" color="success" />
                <Typography variant="h6">Traffic</Typography>
                <Typography>
                  {traffic
                    ? `${traffic.flowSegmentData.currentSpeed} km/h`
                    : "Loading..."}
                </Typography>
              </StyledCard>
            </Grid>
            <Grid item xs={12} sm={4}>
              <StyledCard
                style={{ backgroundColor: "rgba(135, 206, 235, 0.7" }}
              >
                {/* 135, 206, 235 */}
                <LocationOnIcon fontSize="large" color="primary" />
                <Typography variant="h6">Current Location</Typography>
                <Typography>
                  {location
                    ? `${location.latitude.toFixed(
                        2
                      )}, ${location.longitude.toFixed(2)}`
                    : "Loading..."}
                </Typography>
                <Form.Group className="mb-3">
                  <Form.Label style={{ marginRight: 8 }}>Region</Form.Label>
                  <Form.Select onChange={handleRegionChange} required>
                    <option value="">Select a region</option>
                    {regions.map((region) => (
                      <option key={region.name} value={region.name}>
                        {region.name}
                      </option>
                    ))}
                  </Form.Select>
                </Form.Group>
              </StyledCard>
            </Grid>
          </Grid>

          {/* {location && (
          <Box sx={{ marginTop: "2rem", width: "100%", height: "400px" }}>
            <MapContainer
              center={[location.latitude, location.longitude]}
              zoom={13}
              style={{ height: "100%", borderRadius: "12px" }}
            >
              <TileLayer
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                attribution="&copy; OpenStreetMap contributors"
              />
              <Marker position={[location.latitude, location.longitude]}>
                <Popup>Your current location</Popup>
              </Marker>
            </MapContainer>
          </Box>
        )} */}
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "flex-start",
              padding: "20px",
              backgroundColor: "#f5f5f5",
              "background-image":
                "linear-gradient(45deg, rgba(106,17,203,0.3),rgba(37,117,252,0.3))",

              borderRadius: "10px",
              boxShadow: "0 4px 6px rgba(0, 0, 0, 0.1)",
              margin: "10px 0",
              border: "1px,solid orange",
              width: "96.5%",
            }}
          >
            <h2
              style={{
                // fontSize: "24px",
                fontWeight: "bold",
                color: "#2c3e50",
                margin: "0 0 10px",
              }}
            >
              Live Traffic Density Map
            </h2>
            <p
              style={{
                fontSize: "16px",
                color: "rgba(0,0,0,0.8)",
                textAlign: "center",
                margin: "0",
              }}
            >
              Navigate through{" "}
              {selectedRegion?.name ? selectedRegion?.name : ""}’s real-time
              traffic insights to plan your journey efficiently.
            </p>
          </div>

          {location && (
            <Box sx={{ marginTop: "2rem", width: "100%", height: "400px" }}>
              <MapContainer
                center={[location.latitude, location.longitude]}
                zoom={13}
                style={{ height: "100%", borderRadius: "12px" }}
              >
                <ChangeMapView
                  center={[location.latitude, location.longitude]}
                />
                <TileLayer
                  url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                  attribution="&copy; OpenStreetMap contributors"
                />
                <HeatmapLayer data={heatmapData} />
                <Marker position={[location.latitude, location.longitude]}>
                  <Popup>Your current location</Popup>
                </Marker>
              </MapContainer>
            </Box>
          )}
          <TrafficStatsCard trafficData={trafficData} />
          <NewsList>
            <Typography
              variant="h4"
              sx={{ marginBottom: "1rem" }}
              textAlign="center"
            >
              Latest News
            </Typography>
            <Box sx={{ padding: "2rem" }}>
              <Grid container spacing={3}>
                {news.map((article, index) => (
                  <Grid item xs={12} sm={6} md={4} key={index}>
                    <Card
                      sx={{
                        padding: "1.5rem",
                        backgroundColor: "#fff",
                        borderRadius: "8px",
                        boxShadow: "0 4px 10px rgba(0, 0, 0, 0.1)",
                        transition: "transform 0.3s ease, box-shadow 0.3s ease",
                        display: "flex",
                        flexDirection: "column",
                        justifyContent: "space-between",
                        height: "100%",
                        "&:hover": {
                          transform: "scale(1.05)",
                          boxShadow: "0 6px 15px rgba(0, 0, 0, 0.2)",
                        },
                      }}
                    >
                      <Typography variant="h6" gutterBottom>
                        {article.name}
                      </Typography>
                      <Typography
                        variant="body2"
                        color="textSecondary"
                        gutterBottom
                      >
                        {article.description}
                      </Typography>
                      <Typography
                        variant="body2"
                        color="textSecondary"
                        gutterBottom
                      >
                        Sentiment Score: {article.sentimentScore}{" "}
                        {article.sentimentScore > 0
                          ? "(Positive)"
                          : article.sentimentScore < 0
                          ? "(Negative)"
                          : "(Neutral)"}
                      </Typography>
                      <Button
                        variant="contained"
                        color="primary"
                        href={article.url}
                        target="_blank"
                        sx={{
                          marginTop: "1rem",
                          textTransform: "none",
                          borderRadius: "20px",
                          padding: "0.5rem 1.5rem",
                          background:
                            "linear-gradient(45deg, #6a11cb, #2575fc)",
                          color: "#fff",
                          "&:hover": {
                            background:
                              "linear-gradient(45deg, #2575fc, #6a11cb)",
                          },
                        }}
                      >
                        Read More
                      </Button>
                    </Card>
                  </Grid>
                ))}
              </Grid>
            </Box>
          </NewsList>
          <Box sx={{ p: 3, boxShadow: "#eee" }}>
            {predictedData && (
              <div>
                <div style={{ display: "flex", flexDirection: "row" }}>
                  <h1 className="gradient-text">Traffic Predictions </h1>
                  <Lottie
                    animationData={aibot}
                    loop
                    style={{
                      right: 6,
                      top: 6,
                      width: 60,
                      height: 60,
                      left: 10,
                    }}
                  />
                </div>
                <Grid container spacing={3}>
                  {Object.entries(predictedData).map(
                    ([interval, prediction]) => (
                      <Grid item xs={12} sm={6} md={4} key={interval}>
                        <Card sx={{ borderRadius: 2, boxShadow: 3 }}>
                          <CardContent>
                            <Typography
                              variant="h6"
                              align="center"
                              gutterBottom
                              sx={{
                                textTransform: "uppercase",
                                fontWeight: "bold",
                              }}
                            >
                              {interval.replace("_", " ")} Prediction
                            </Typography>
                            <Box
                              sx={{
                                display: "flex",
                                justifyContent: "center",
                                mb: 2,
                              }}
                            >
                              <Chip
                                label={prediction.category}
                                color={getBadgeVariant(prediction.category)}
                                sx={{
                                  fontSize: "0.85rem",
                                  fontWeight: "bold",
                                }}
                              />
                            </Box>
                            <Typography variant="body1" gutterBottom>
                              <strong>Volume:</strong> {prediction.volume}
                              {" k"}
                              vehicles/hour
                            </Typography>
                            <Typography variant="body2" gutterBottom>
                              <strong>Description:</strong>{" "}
                              {prediction.description}
                            </Typography>
                            <Typography variant="body2">
                              <strong>Recommendation:</strong>{" "}
                              {prediction.recommendation}
                            </Typography>
                          </CardContent>
                        </Card>
                      </Grid>
                    )
                  )}
                </Grid>
              </div>
            )}
          </Box>
        </DashboardContainer>
      </>
    </ErrorBoundary>
  );
};

export default Dashboard;
