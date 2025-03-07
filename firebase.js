import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";
import { getDatabase, ref, set, onValue, remove } from "firebase/database";
import { getStorage } from "firebase/storage";

import { getFirestore, doc, setDoc, collection, deleteDoc, onSnapshot } from "firebase/firestore";

const firebaseConfig = {
  apiKey: "AIzaSyD83jUZiPHrdppBmxLsRlo4O60FJTDgIyk",
  authDomain: "calloc-landing-website.firebaseapp.com",
  databaseURL: "https://calloc-landing-website-default-rtdb.firebaseio.com",
  projectId: "calloc-landing-website",
  storageBucket: "calloc-landing-website.appspot.com",
  messagingSenderId: "985509226560",
  appId: "1:985509226560:web:a1165c2f58ce615a2f17a5",
  measurementId: "G-S1GL036FM5"
};

const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);
export const db = getFirestore(app);
export const storage = getStorage(app);

