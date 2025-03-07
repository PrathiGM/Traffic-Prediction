import React, { useEffect, useState } from "react";
import {
  GoogleMap,
  DirectionsRenderer,
  TrafficLayer,
  useJsApiLoader,
  Marker,
  DirectionsService,
  Polyline,
  Autocomplete,
} from "@react-google-maps/api";
import {
  Container,
  Grid,
  Button,
  Typography,
  CircularProgress,
  Paper,
  List,
  ListItem,
  ListItemText,
  Drawer,
  IconButton,
  AppBar,
  Box,
  Toolbar,
  Snackbar,
  ListItemIcon,
  Card,
  CardContent,
  Chip,
} from "@mui/material";
import {
  getFirestore,
  doc,
  setDoc,
  collection,
  deleteDoc,
  onSnapshot,
  getDocs,
} from "firebase/firestore";
import axios from "axios";
import { styled } from "@mui/system";
import { useNavigate } from "react-router-dom";
import MenuIcon from "@mui/icons-material/Menu";
import SwapHorizIcon from "@mui/icons-material/SwapHoriz";
import { db } from "./firebase";
import MuiAlert from "@mui/material/Alert";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import ErrorIcon from "@mui/icons-material/Error";
import DirectionsIcon from "@mui/icons-material/Directions";
import Sentiment from "sentiment";
import ErrorBoundary from "./ErrorBoundary";
import Lottie from "lottie-react";
import aibot from "./components/aijson.json";

const Alert = React.forwardRef(function Alert(props, ref) {
  return <MuiAlert elevation={6} ref={ref} variant="filled" {...props} />;
});

// import { Place, Flag } from "@mui/icons-material"; // Importing icons
const containerStyle = {
  width: "100%",
  height: "600px",
};
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

const center = { lat: 12.9716, lng: 77.5946 }; // Default center (Bangalore)

const RoutePlanner = () => {
  const [startPoint, setStartPoint] = useState(null);
  const [endPoint, setEndPoint] = useState(null);
  const [optimalRouteIndex, setOptimalRouteIndex] = useState(null);
  const [loading, setLoading] = useState(false);
  const [trafficVisible, setTrafficVisible] = useState(true);
  const [weatherInfo, setWeatherInfo] = useState(null);
  const [news, setNews] = useState([]);
  const [routeDetails, setRouteDetails] = useState([]);
  const [startPointAddress, setStartPointAddress] = useState();
  const [endPointAddress, setEndPointAddress] = useState();
  const [blockedRoutes, setBlockedRoutes] = useState([]);
  const [directionsResult, setDirectionsResult] = useState(null);
  const [optimalRoute, setOptimalRoute] = useState(null);
  const [isFirstTime, setisFirstTime] = useState(true);
  const [show, setShow] = useState(false);
  const [updateBlocked, setUpdateBlocked] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [autocompleteStart, setAutocompleteStart] = useState(null);
  const [autocompleteEnd, setAutocompleteEnd] = useState(null);
  const [alertOpen, setAlertOpen] = useState(false);
  const [directions, setDirections] = useState(null);
  const [selectedRouteIndex, setSelectedRouteIndex] = useState(null);
  const [routePolylines, setRoutePolylines] = useState([]);
  const sentiment = new Sentiment();
  const navigate = useNavigate();

  const { isLoaded } = useJsApiLoader({
    googleMapsApiKey: "AIzaSyD590z__itIHB85Rrz0XJxEpi-PVYPs2b0",
    libraries: [
      "places",
      "geocoding",
      "maps",
      "marker",
      "routes",
      "streetView",
      "core",
      "visualization",
      "geometry",
    ],
  });
  const handleAlertClose = (event, reason) => {
    if (reason === "clickaway") {
      return;
    }
    setAlertOpen(false);
  };
  const decodePolyline = (polyline) => {
    const coordinates = [];
    let index = 0,
      len = polyline.length;
    let lat = 0,
      lng = 0;

    while (index < len) {
      let b,
        shift = 0,
        result = 0;
      do {
        b = polyline.charCodeAt(index++) - 63;
        result |= (b & 0x1f) << shift;
        shift += 5;
      } while (b >= 0x20);
      const dlat = result & 1 ? ~(result >> 1) : result >> 1;
      lat += dlat;

      shift = 0;
      result = 0;
      do {
        b = polyline.charCodeAt(index++) - 63;
        result |= (b & 0x1f) << shift;
        shift += 5;
      } while (b >= 0x20);
      const dlng = result & 1 ? ~(result >> 1) : result >> 1;
      lng += dlng;

      coordinates.push({ lat: lat / 1e5, lng: lng / 1e5 });
    }

    return coordinates;
  };
  useEffect(() => {
    const fetchBlockedRoutes = async () => {
      try {
        const blockedRoutesCollection = collection(db, "blockedRoutes");
        const querySnapshot = await getDocs(blockedRoutesCollection);
        const routes = [];
        querySnapshot.forEach((doc) => {
          const data = doc.data();
          const polyline = decodePolyline(data.polyline); // Decode polyline to coordinates
          routes.push(polyline);
        });

        console.log("blocked routes now", routes);
        setBlockedRoutes(routes);
      } catch (error) {
        console.error("Error fetching blocked routes: ", error);
      }
    };

    fetchBlockedRoutes();
  }, []);
  useEffect(() => {
    const fetchPredictions = async () => {
      const trafficResponse = await axios.get(
        `https://api.tomtom.com/traffic/services/4/flowSegmentData/relative0/10/json?point=${endPoint.lat},${endPoint.lng}&key=5QpZLwcTD1Gz8dJ0O7o1u9vGokfRF1Og`
      );
      const trafficDataw = trafficResponse.data.flowSegmentData;

      // Generate a `currentTrafficData` array dynamically from the TomTom data
      // Example: [currentSpeed, freeFlowSpeed, confidence] as mock data points
      const currentTrafficData = [
        trafficDataw.currentSpeed,
        trafficDataw.freeFlowSpeed,
        Math.round(trafficDataw.confidence * 100), // Convert confidence to percentage
      ];
      const endPlaceName = await geocodeLatLng(endPoint.lat, endPoint.lng);
      const response = await axios.post("http://localhost:5000/predict_lstm", {
        region: endPlaceName?.localityName,
        latitude: endPoint.lat,
        longitude: endPoint.lng,
        weatherData: JSON.stringify({
          temperature: (weatherInfo.start.main.temp - 273.15).toFixed(2), // Convert Kelvin to Celsius
          condition: weatherInfo.start.weather[0].description,
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
    if (endPoint && weatherInfo) {
      fetchPredictions();
    }
  }, [endPoint, weatherInfo]);
  // useEffect(() => {
  //   if (blockedRoutes && blockedRoutes?.length > 0) {
  //     setUpdateBlocked(!updateBlocked);
  //   }
  // }, [blockedRoutes]);
  const [userData, setuserData] = useState();
  const [predictedData, setPredictedData] = useState(null);

  const getUserData = () => {
    const userData = localStorage.getItem("user");
    return userData ? JSON.parse(userData) : null;
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

  useEffect(() => {
    setuserData(getUserData());
  }, []);
  if (!isLoaded) return <CircularProgress />;
  const fetchBlockedRoutesnew = async () => {
    try {
      const blockedRoutesCollection = collection(db, "blockedRoutes");
      const querySnapshot = await getDocs(blockedRoutesCollection);
      const routes = [];
      querySnapshot.forEach((doc) => {
        const data = doc.data();
        const polyline = decodePolyline(data.polyline); // Decode polyline to coordinates
        routes.push(polyline);
      });

      console.log("blocked routes now", routes);
      setBlockedRoutes(routes);
    } catch (error) {
      console.error("Error fetching blocked routes: ", error);
    }
  };
  // const fetchRoutes = async () => {
  //   if (!startPoint || !endPoint) {
  //     alert("Please select both start and end points!");
  //     return;
  //   }

  //   setLoading(true);

  //   const directionsService = new google.maps.DirectionsService();

  //   directionsService.route(
  //     {
  //       origin: startPoint,
  //       destination: endPoint,
  //       travelMode: google.maps.TravelMode.DRIVING,
  //       provideRouteAlternatives: true,
  //     },
  //     (result, status) => {
  //       if (status === "OK") {
  //         setDirections(result);
  //         determineBestRoute(result.routes);
  //         fetchRouteTrafficDetails(result.routes);
  //         console.log("routes generated", result);
  //       } else {
  //         alert("Error fetching directions: " + status);
  //       }
  //       setLoading(false);
  //       setShow(true);
  //     }
  //   );
  // };
  // Helper function to safely get lat/lng values
  const getLatLng = (point) => {
    if (!point) return null;

    // Handle LatLng object
    if (point instanceof google.maps.LatLng) {
      return {
        lat: point.lat(),
        lng: point.lng(),
      };
    }

    // Handle LatLngLiteral object
    if (point.lat && point.lng) {
      return {
        lat: typeof point.lat === "function" ? point.lat() : point.lat,
        lng: typeof point.lng === "function" ? point.lng() : point.lng,
      };
    }

    return null;
  };

  // Updated isRouteSimilar function
  const isRouteSimilar = (route1, route2, threshold = 0.8) => {
    if (!route1 || !route2) return false;

    const path1 = route1.overview_path || [];
    const path2 = route2.overview_path || [];

    // Compare route distance and duration
    if (route1.legs?.[0] && route2.legs?.[0]) {
      const distanceDiff = Math.abs(
        route1.legs[0].distance.value - route2.legs[0].distance.value
      );
      const durationDiff = Math.abs(
        route1.legs[0].duration.value - route2.legs[0].duration.value
      );

      // If routes are significantly different in distance or duration, consider them unique
      if (distanceDiff > 1000 || durationDiff > 300) {
        return false;
      }
    }

    // Compare points along the path
    let similarPoints = 0;
    const pointsToCheck = Math.min(path1.length, path2.length, 10);

    if (pointsToCheck === 0) return false;

    for (let i = 0; i < pointsToCheck; i++) {
      const index1 = Math.floor((i / pointsToCheck) * path1.length);
      const index2 = Math.floor((i / pointsToCheck) * path2.length);

      const point1 = path1[index1];
      const point2 = path2[index2];

      if (!point1 || !point2) continue;

      try {
        const latLng1 = new google.maps.LatLng(getLatLng(point1));
        const latLng2 = new google.maps.LatLng(getLatLng(point2));

        const distance = google.maps.geometry.spherical.computeDistanceBetween(
          latLng1,
          latLng2
        );
        if (distance < 500) {
          // 500 meters threshold
          similarPoints++;
        }
      } catch (error) {
        console.warn("Error comparing points:", error);
        continue;
      }
    }

    return similarPoints / pointsToCheck > threshold;
  };

  // Updated fetchMultipleRoutes function
  const fetchMultipleRoutes = async () => {
    if (!startPoint || !endPoint) {
      alert("Please select both start and end points!");
      return;
    }

    setLoading(true);
    const directionsService = new google.maps.DirectionsService();
    let allRoutes = [];

    // Convert startPoint and endPoint to LatLng objects
    const origin = new google.maps.LatLng(startPoint.lat, startPoint.lng);
    const destination = new google.maps.LatLng(endPoint.lat, endPoint.lng);

    const waypointCombinations = [
      [],
      [
        {
          location: new google.maps.LatLng(
            startPoint.lat + 0.01,
            startPoint.lng + 0.01
          ),
          stopover: false,
        },
      ],
      [
        {
          location: new google.maps.LatLng(
            startPoint.lat - 0.01,
            startPoint.lng - 0.01
          ),
          stopover: false,
        },
      ],
      [
        {
          location: new google.maps.LatLng(
            endPoint.lat + 0.01,
            endPoint.lng + 0.01
          ),
          stopover: false,
        },
      ],
      [
        {
          location: new google.maps.LatLng(
            endPoint.lat - 0.01,
            endPoint.lng - 0.01
          ),
          stopover: false,
        },
      ],
    ];

    try {
      const routePromises = waypointCombinations.map((waypoints) => {
        return new Promise((resolve, reject) => {
          directionsService.route(
            {
              origin: origin,
              destination: destination,
              travelMode: google.maps.TravelMode.DRIVING,
              provideRouteAlternatives: true,
              optimizeWaypoints: true,
              waypoints: waypoints,
            },
            (result, status) => {
              if (status === "OK") {
                resolve(result.routes);
              } else {
                console.warn("Route calculation failed:", status);
                resolve([]);
              }
            }
          );
        });
      });

      const results = await Promise.all(routePromises);

      // Process and store routes
      const processedRoutes = results.flat().filter((route) => route !== null);

      // Filter unique routes
      const uniqueRoutes = [];
      for (const route of processedRoutes) {
        const isDuplicate = uniqueRoutes.some((existingRoute) =>
          isRouteSimilar(route, existingRoute)
        );
        if (!isDuplicate) {
          uniqueRoutes.push(route);
        }
      }

      if (uniqueRoutes.length > 0) {
        setDirections({
          ...results[0],
          routes: uniqueRoutes,
        });

        // Extract and store polylines for each route
        const polylines = uniqueRoutes.map((route) =>
          route.overview_path.map((point) => getLatLng(point))
        );
        setRoutePolylines(polylines);

        determineBestRoute(uniqueRoutes);
        fetchRouteTrafficDetails(uniqueRoutes);
        setSelectedRouteIndex(0);
      } else {
        alert("No valid routes found. Please try different points.");
      }
    } catch (error) {
      console.error("Error fetching routes:", error);
      alert("Error calculating routes. Please try again.");
    } finally {
      setLoading(false);
      setShow(true);
    }
  };

  const determineBestRoute = (routes) => {
    // Determine the shortest-duration route
    let bestIndex = 0;
    let minDuration = routes[0].legs[0].duration.value;

    routes.forEach((route, index) => {
      if (route.legs[0].duration.value < minDuration) {
        bestIndex = index;
        minDuration = route.legs[0].duration.value;
      }
    });

    setOptimalRouteIndex(bestIndex);
  };

  const fetchRouteTrafficDetails = (routes) => {
    // Collect traffic data and additional details for each route
    const details = routes.map((route, index) => {
      const leg = route.legs[0]; // Assume only one leg for A -> B
      return {
        routeIndex: index,
        distance: leg.distance.text,
        routedata: route,
        duration: leg.duration.text,
        trafficSpeed: Math.floor(Math.random() * (80 - 30) + 30), // Mock: random speed
        trafficVolume: Math.floor(Math.random() * (100 - 50) + 50), // Mock: random volume
        estimatedArrivalTime: new Date(
          Date.now() + leg.duration.value * 1000
        ).toLocaleTimeString(), // Add duration to current time
      };
    });

    setRouteDetails(details);
  };

  const geocodeLatLng = async (lat, lng) => {
    const geocodeApiKey = "AIzaSyD590z__itIHB85Rrz0XJxEpi-PVYPs2b0"; // Your Google Maps API Key
    try {
      const response = await axios.get(
        `https://maps.googleapis.com/maps/api/geocode/json?latlng=${lat},${lng}&key=${geocodeApiKey}`
      );

      if (response.data.status !== "OK" || !response.data.results.length) {
        throw new Error("No results found");
      }

      // Extract the complete address
      const completeAddress =
        response.data.results[0]?.formatted_address || "Unknown address";

      // Extract locality name
      const localityComponent =
        response.data.results[0]?.address_components.find((comp) =>
          comp.types.includes("locality")
        );
      const localityName = localityComponent?.long_name || "Unknown locality";

      return {
        localityName,
        completeAddress,
      };
    } catch (error) {
      console.error("Error during geocoding:", error);
      return {
        localityName: "Unknown locality",
        completeAddress: "Unknown address",
      };
    }
  };

  const fetchWeatherAndNews = async (endPoint) => {
    const openWeatherApiKey = "4b3120747b9ef1f222316cf9cf47bd7a";
    const NewsApiKey = "a70199da760347429bf6df19030da50f";

    try {
      const startPlaceName = await geocodeLatLng(
        startPoint.lat,
        startPoint.lng
      );
      const endPlaceName = await geocodeLatLng(endPoint.lat, endPoint.lng);
      setStartPointAddress(startPlaceName?.completeAddress);
      setEndPointAddress(endPlaceName?.completeAddress);
      // Fetch weather data for start and end points
      const weatherStart = axios.get(
        `https://api.openweathermap.org/data/2.5/weather?lat=${startPoint.lat}&lon=${startPoint.lng}&appid=${openWeatherApiKey}`
      );

      const weatherEnd = axios.get(
        `https://api.openweathermap.org/data/2.5/weather?lat=${endPoint.lat}&lon=${endPoint.lng}&appid=${openWeatherApiKey}`
      );

      const [startWeather, endWeather] = await Promise.all([
        weatherStart,
        weatherEnd,
      ]);

      setWeatherInfo({
        start: startWeather.data,
        end: endWeather.data,
      });

      // Fetch news data for the selected region
      const newsResponseStart = await axios.get(
        `https://api.bing.microsoft.com/v7.0/news/search?q=traffic news and road blockages in ${startPlaceName?.localityName}&count=5`,
        {
          headers: { "Ocp-Apim-Subscription-Key": bingNewsApiKey },
        }
      );
      const newsResponseEnd = await axios.get(
        `https://api.bing.microsoft.com/v7.0/news/search?q=traffic news and road blockages in ${endPlaceName?.localityName}&count=5`,
        {
          headers: { "Ocp-Apim-Subscription-Key": bingNewsApiKey },
        }
      );

      const newsResponseStartSentiment = newsResponseStart.data.value.map(
        (item) => {
          return {
            ...item,
            sentimentScore: sentiment.analyze(item.description).score,
            sentimentComparative: sentiment.analyze(item.description)
              .comparative,
          };
        }
      );

      const newsResponseEndSentiment = newsResponseEnd.data.value.map(
        (item) => {
          return {
            ...item,
            sentimentScore: sentiment.analyze(item.description).score,
            sentimentComparative: sentiment.analyze(item.description)
              .comparative,
          };
        }
      );

      setNews([...newsResponseStartSentiment, ...newsResponseEndSentiment]);
    } catch (error) {
      console.error("Error fetching data:", error);
    }
  };

  const handleMapClick = (event) => {
    if (!startPoint) {
      setStartPoint({
        lat: event.latLng.lat(),
        lng: event.latLng.lng(),
      });
    } else if (!endPoint) {
      const selectedEndPoint = {
        lat: event.latLng.lat(),
        lng: event.latLng.lng(),
      };
      setEndPoint(selectedEndPoint);
      fetchWeatherAndNews(selectedEndPoint);
      // Automatically fetch weather and news
    }
  };

  const handleRouteSelect = (index) => {
    if (!directions || !directions.routes[index]) {
      console.error("Invalid route selection");
      return;
    }

    const selectedRoute = directions.routes[index];
    const isBlocked = checkRouteBlockedNew(selectedRoute.overview_path);

    if (isBlocked) {
      setAlertOpen(true);
      return;
    }

    setSelectedRouteIndex(index);
  };
  // Decode polyline string into coordinates

  const directionsCallback = (response) => {
    if (response && response.status === "OK") {
      setDirectionsResult(response);
    }
  };

  const checkRouteBlocked = (route) => {
    for (const blockedRoute of blockedRoutes) {
      console.log("blocked", blockedRoutes);
      for (const point of route) {
        if (
          blockedRoute.some(
            (blockedPoint) =>
              Math.abs(blockedPoint.lat - point.lat) < 0.001 &&
              Math.abs(blockedPoint.lng - point.lng) < 0.001
          )
        ) {
          return true; // Route is blocked
        }
      }
    }
    return false; // Route is not blocked
  };

  const checkRouteBlockedNew = (route) => {
    console.log("Checking Route:", route);

    // Loop through the list of blocked routes
    for (const blockedRoute of blockedRoutes) {
      console.log("Checking against Blocked Route:", blockedRoute);

      // Check if any point in the current route matches any point in the blocked route
      const isBlocked = route.some((point) => {
        // Extract the numeric values for lat and lng
        const pointLat =
          typeof point.lat === "function" ? point.lat() : point.lat;
        const pointLng =
          typeof point.lng === "function" ? point.lng() : point.lng;

        return blockedRoute.some((blockedPoint) => {
          const blockedLat =
            typeof blockedPoint.lat === "function"
              ? blockedPoint.lat()
              : blockedPoint.lat;
          const blockedLng =
            typeof blockedPoint.lng === "function"
              ? blockedPoint.lng()
              : blockedPoint.lng;

          const isMatch =
            Math.abs(blockedLat - pointLat) < 0.0005 &&
            Math.abs(blockedLng - pointLng) < 0.0005;

          if (isMatch) {
            console.log("Matching Point Found:", point, blockedPoint);
          }
          return isMatch;
        });
      });

      if (isBlocked) {
        // setUpdateBlocked(true);
        console.log("Route Blocked Detected!");
        return true;
      }
    }

    console.log("No Blockage Detected");
    return false;
  };
  const handleStartPlaceSelected = () => {
    if (autocompleteStart !== null) {
      const place = autocompleteStart.getPlace();
      if (place.geometry) {
        const location = place.geometry.location;
        setStartPoint({
          lat: location.lat(),
          lng: location.lng(),
        });
      } else {
        alert("No geometry found for this place.");
      }
    }
  };
  const handleEndPlaceSelected = () => {
    if (autocompleteEnd !== null) {
      const place = autocompleteEnd.getPlace();
      if (place.geometry) {
        const location = place.geometry.location;
        setEndPoint({
          lat: location.lat(),
          lng: location.lng(),
        });
      } else {
        alert("No geometry found for this place.");
      }
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
              Traffix Route Planner
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
          <Container>
            <Snackbar
              open={alertOpen}
              autoHideDuration={3000} // Alert will disappear after 3 seconds
              onClose={handleAlertClose}
              anchorOrigin={{ vertical: "top", horizontal: "center" }} // Position of the alert
            >
              <Alert onClose={handleAlertClose} severity="warning">
                Route is blocked! Please choose a different path.
              </Alert>
            </Snackbar>
            {/* <Typography variant="h4" textAlign="center" gutterBottom>
            Route Planner with Traffic, Weather, and News Insights
          </Typography> */}
            <h1
              className="gradient-text"
              style={{ margin: 0, alignSelf: "center", textAlign: "center" }}
            >
              Route Planner with Traffic, Weather, and News Insights
            </h1>
            <div
              style={{
                display: "flex",
                flexDirection: "row",
                justifyContent: "space-between",
              }}
            >
              <div>
                <h4>Select Start Place</h4>
                <Autocomplete
                  onLoad={(autoC) => setAutocompleteStart(autoC)}
                  onPlaceChanged={handleStartPlaceSelected}
                >
                  <input
                    type="text"
                    placeholder="Search for a start place"
                    style={{
                      width: "300px",
                      padding: "10px",
                      marginBottom: "10px",
                    }}
                  />
                </Autocomplete>
              </div>
              <SwapHorizIcon
                style={{ alignSelf: "center", width: "100px", height: "60px" }}
              />
              <div>
                <h4>Select End Place</h4>
                <Autocomplete
                  onLoad={(autoC) => setAutocompleteEnd(autoC)}
                  onPlaceChanged={handleEndPlaceSelected}
                >
                  <input
                    type="text"
                    placeholder="Search for an end place"
                    style={{
                      width: "300px",
                      padding: "10px",
                      marginBottom: "10px",
                    }}
                  />
                </Autocomplete>
              </div>
            </div>
            <Grid container spacing={2}>
              <Grid item xs={12}>
                <GoogleMap
                  mapContainerStyle={containerStyle}
                  center={center}
                  zoom={13}
                  onClick={handleMapClick}
                >
                  {startPoint && <Marker position={startPoint} label="Start" />}
                  {endPoint && <Marker position={endPoint} label="End" />}
                  {blockedRoutes.map((route, index) => (
                    <Polyline
                      key={index}
                      path={route}
                      options={{
                        strokeColor: "#000",
                        strokeOpacity: 0.8,
                        strokeWeight: 4,
                      }}
                    />
                  ))}
                  {routePolylines.map((polyline, index) => (
                    <Polyline
                      key={`route-${index}`}
                      path={polyline}
                      options={{
                        strokeColor:
                          index === selectedRouteIndex ? "#4CAF50" : "#2196F3",
                        strokeWeight: index === selectedRouteIndex ? 5 : 3,
                        strokeOpacity: index === selectedRouteIndex ? 0.8 : 0.4,
                        visible: true,
                      }}
                      onClick={() => handleRouteSelect(index)}
                    />
                  ))}
                  {directions && (
                    <DirectionsRenderer
                      directions={directions}
                      routeIndex={selectedRouteIndex}
                      options={{
                        suppressMarkers: false,
                        suppressPolylines: false,
                        polylineOptions: {
                          strokeColor:
                            selectedRouteIndex === optimalRouteIndex
                              ? "#4CAF50"
                              : "#2196F3",
                          strokeWeight: 5,
                          strokeOpacity: 0.8,
                        },
                      }}
                    />
                  )}
                  {trafficVisible && <TrafficLayer />}
                </GoogleMap>
              </Grid>
              <Grid item xs={12} md={6}>
                {/* <Button
                variant="contained"
                color="primary"
                fullWidth
                onClick={fetchRoutes}
                disabled={loading}
              >
                {loading ? "Calculating Routes..." : "Fetch Optimal Routes"}
              </Button> */}
                <Button
                  variant="contained"
                  fullWidth
                  onClick={fetchMultipleRoutes}
                  disabled={loading}
                  style={{
                    background: loading
                      ? "#a5a5a5"
                      : "linear-gradient(45deg, #673ab7, #512da8)",
                    color: "#fff",
                    padding: "12px 20px",
                    fontSize: "16px",
                    fontWeight: "bold",
                    textTransform: "capitalize",
                    boxShadow: loading
                      ? "none"
                      : "0 4px 10px rgba(0, 0, 0, 0.25), 0 1px 3px rgba(0, 0, 0, 0.15)",
                    borderRadius: "8px",
                    transition: "all 0.3s ease",
                    cursor: loading ? "not-allowed" : "pointer",
                  }}
                >
                  {loading ? (
                    <span
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: "8px",
                      }}
                    >
                      <CircularProgress size={20} style={{ color: "#fff" }} />{" "}
                      Calculating Routes...
                    </span>
                  ) : (
                    "Fetch Optimal Routes"
                  )}
                </Button>
              </Grid>
            </Grid>

            {routeDetails.length > 0 && (
              <Paper style={{ padding: "10px", marginTop: "20px" }}>
                <Typography variant="h6">Route Details🛣️🛣️:</Typography>

                <List>
                  {routeDetails.map((detail, index) => {
                    const isBlocked = checkRouteBlockedNew(
                      detail.routedata.overview_path
                    );
                    const isOptimal = detail?.routeIndex === optimalRouteIndex;
                    const isSelected =
                      detail?.routeIndex === selectedRouteIndex;

                    return (
                      <ListItem
                        key={detail?.routeIndex}
                        button
                        selected={isSelected}
                        onClick={() => {
                          if (isBlocked) {
                            setAlertOpen(true);
                          } else {
                            console.log("selectedindex", detail?.routeIndex);
                            handleRouteSelect(detail?.routeIndex);
                          }
                        }}
                        style={{
                          border: isSelected
                            ? "2px solid #2196F3"
                            : isOptimal
                            ? "2px solid #4CAF50"
                            : "1px solid #e0e0e0",
                          marginBottom: "10px",
                          borderRadius: "8px",
                          backgroundColor: isSelected
                            ? "#e3f2fd"
                            : isOptimal
                            ? "#f1f8e9"
                            : "#ffffff",
                          transform: isSelected ? "scale(1.02)" : "scale(1)",
                          transition: "all 0.2s ease-in-out",
                          boxShadow: isSelected
                            ? "0 2px 8px rgba(33, 150, 243, 0.2)"
                            : "none",
                        }}
                      >
                        <ListItemIcon>
                          {isBlocked ? (
                            <ErrorIcon style={{ color: "#f44336" }} />
                          ) : isSelected ? (
                            <DirectionsIcon style={{ color: "#2196F3" }} />
                          ) : isOptimal ? (
                            <CheckCircleIcon style={{ color: "#4CAF50" }} />
                          ) : (
                            <DirectionsIcon style={{ color: "#9e9e9e" }} />
                          )}
                        </ListItemIcon>
                        <ListItemText
                          primary={
                            <Typography
                              style={{
                                fontWeight: isSelected ? "600" : "normal",
                                color: isSelected ? "#2196F3" : "inherit",
                              }}
                            >
                              {`Route ${index + 1} ${
                                isOptimal ? "(Optimal)" : ""
                              } ${isSelected ? "(Selected)" : ""}`}
                            </Typography>
                          }
                          secondary={
                            <Typography
                              style={{
                                color: isBlocked
                                  ? "#f44336"
                                  : isSelected
                                  ? "#1976d2"
                                  : "inherit",
                              }}
                            >
                              {`Distance: ${detail.distance}, Duration: ${
                                detail.duration
                              }, Traffic Speed: ${
                                detail.trafficSpeed
                              } km/h, Traffic Volume: ${
                                detail.trafficVolume
                              }, ETA: ${detail.estimatedArrivalTime}
                  ${isBlocked ? " (Blocked)" : ""}`}
                            </Typography>
                          }
                        />
                      </ListItem>
                    );
                  })}
                </List>
              </Paper>
            )}

            {weatherInfo && (
              <Paper
                style={{
                  padding: "20px",
                  marginTop: "20px",
                  backgroundColor: "#f5f5f5",
                  borderRadius: "10px",
                  boxShadow: "0 4px 8px rgba(0, 0, 0, 0.1)",
                }}
              >
                <Typography
                  variant="h5"
                  style={{
                    marginBottom: "15px",
                    fontWeight: "bold",
                    color: "#3f51b5",
                    textAlign: "center",
                  }}
                >
                  Weather Information☀️⛄
                </Typography>
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    flexWrap: "wrap",
                    gap: "15px",
                  }}
                >
                  {/* Start Point */}
                  <div
                    style={{
                      flex: "1 1 45%",
                      textAlign: "center",
                      display: "flex",
                      flexDirection: "column",
                      alignItems: "center",
                    }}
                  >
                    {/* <Place style={{ fontSize: "40px", color: "#1976d2" }} /> */}
                    {/* Start Icon */}
                    <Typography
                      variant="subtitle1"
                      style={{ fontWeight: "bold" }}
                    >
                      {startPointAddress}
                    </Typography>
                    <Typography
                      variant="body1"
                      style={{
                        backgroundColor: "#e3f2fd",
                        padding: "10px",
                        borderRadius: "5px",
                        marginTop: "5px",
                      }}
                    >
                      {weatherInfo.start.weather[0].description}
                    </Typography>
                  </div>

                  {/* End Point */}
                  <div
                    style={{
                      flex: "1 1 45%",
                      textAlign: "center",
                      display: "flex",
                      flexDirection: "column",
                      alignItems: "center",
                    }}
                  >
                    {/* <Flag style={{ fontSize: "40px", color: "#43a047" }} /> */}
                    {/* End Icon */}
                    <Typography
                      variant="subtitle1"
                      style={{ fontWeight: "bold" }}
                    >
                      {endPointAddress}
                    </Typography>
                    <Typography
                      variant="body1"
                      style={{
                        backgroundColor: "#e8f5e9",
                        padding: "10px",
                        borderRadius: "5px",
                        marginTop: "5px",
                      }}
                    >
                      {weatherInfo.end.weather[0].description}
                    </Typography>
                  </div>
                </div>
              </Paper>
            )}

            <div
              style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "flex-start",
                padding: "20px",
                backgroundColor: "#f5f5f5",
                borderRadius: "10px",
                boxShadow: "0 4px 6px rgba(0, 0, 0, 0.1)",
                margin: "10px 0",
              }}
            >
              <h2
                style={{
                  fontSize: "24px",
                  fontWeight: "bold",
                  color: "#2c3e50",
                  margin: "0 0 10px",
                }}
              >
                Recent Updates Near You📰📰
              </h2>
            </div>
            <ul>
              {news.map((article, index) => (
                <li key={index}>
                  <div style={{ fontSize: 14, padding: "5px 0" }}>
                    <strong>{article.name}</strong>
                    <br />
                    Sentiment Score: {article.sentimentScore}{" "}
                    {article.sentimentScore > 0
                      ? "(Positive)"
                      : article.sentimentScore < 0
                      ? "(Negative)"
                      : "(Neutral)"}
                  </div>
                </li>
              ))}
            </ul>
            <Box sx={{ p: 3 }}>
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
          </Container>
        </DashboardContainer>
      </>
    </ErrorBoundary>
  );
};

export default RoutePlanner;
