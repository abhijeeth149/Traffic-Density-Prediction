import { useState } from "react";

// Predefined data for cities, areas, and days
const cityAreas = {
  Hyderabad: ["Alwal", "Charminar", "Begumpet", "Patancheru", "Attapur", "Shamshabad", "Nizampet", "Gachibowli", "Chandanagar", "Himayatnagar"],
  Mumbai: ["Girgaon", "Matunga", "Mazgaon", "Juhu Beach", "Prabhadevi", "Andheri West", "Chembur", "Dadar East", "Kalbadevi", "Lower Parel"],
  Bengaluru: ["Vijayanagar", "Hebbal", "Nagavara", "Koramangala", "Majestic", "MG Road", "Hulimavu", "Yelahanka", "Ulsoor", "Jayanagar"],
  Chennai: ["Mylapore", "Anna Nagar", "Adyar", "Nungambakkam", "Mandaveli", "Chromepet", "Choolaimedu", "Guindy", "T Nagar", "Royapettah"],
  Kolkata: ["Cossipore", "Esplanade", "Joka", "Salt Lake", "Rajarhat", "Sealdah", "Tangra", "Dum Dum", "Alipore", "Maniktala"],
};

const daysOfWeek = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];

export default function HomePage() {
  // State variables for form inputs
  const [selectedCity, setSelectedCity] = useState("");
  const [selectedArea, setSelectedArea] = useState("");
  const [selectedDay, setSelectedDay] = useState("");
  const [time, setTime] = useState("");

  // Handle form submission
  const handleSubmit = () => {
    alert(`Predicting traffic for ${selectedArea}, ${selectedCity} on ${selectedDay} at ${time}`);
    // You can integrate an API call here to fetch traffic predictions
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100 p-4">
      <div className="bg-white shadow-lg rounded-lg p-6 w-full max-w-md">
        {/* Page Title */}
        <h1 className="text-2xl font-bold mb-4 text-center">Urban Traffic Density Prediction</h1>

        {/* City Selection */}
        <label className="block mb-2">Select City</label>
        <select
          className="w-full p-2 border rounded mb-4"
          value={selectedCity}
          onChange={(e) => {
            setSelectedCity(e.target.value);
            setSelectedArea(""); // Reset area when city changes
          }}
        >
          <option value="" disabled>Select a city</option>
          {Object.keys(cityAreas).map((city) => (
            <option key={city} value={city}>{city}</option>
          ))}
        </select>

        {/* Area Selection (only visible if a city is selected) */}
        {selectedCity && (
          <>
            <label className="block mb-2">Select Area</label>
            <select
              className="w-full p-2 border rounded mb-4"
              value={selectedArea}
              onChange={(e) => setSelectedArea(e.target.value)}
            >
              <option value="" disabled>Select an area</option>
              {cityAreas[selectedCity].map((area) => (
                <option key={area} value={area}>{area}</option>
              ))}
            </select>
          </>
        )}

        {/* Day Selection */}
        <label className="block mb-2">Select Day</label>
        <select
          className="w-full p-2 border rounded mb-4"
          value={selectedDay}
          onChange={(e) => setSelectedDay(e.target.value)}
        >
          <option value="" disabled>Select a day</option>
          {daysOfWeek.map((day) => (
            <option key={day} value={day}>{day}</option>
          ))}
        </select>

        {/* Time Selection */}
        <label className="block mb-2">Select Time</label>
        <input
          type="time"
          className="w-full p-2 border rounded mb-4"
          value={time}
          onChange={(e) => setTime(e.target.value)}
        />

        {/* Submit Button */}
        <button
          className="w-full bg-blue-500 text-white p-2 rounded hover:bg-blue-600 disabled:bg-gray-400"
          onClick={handleSubmit}
          disabled={!selectedCity || !selectedArea || !selectedDay || !time}
        >
          Proceed
        </button>
      </div>
    </div>
  );
}