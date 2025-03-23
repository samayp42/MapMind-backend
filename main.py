import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from dotenv import load_dotenv
import json
import requests
import math
from urllib.parse import quote

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-2.0-flash-thinking-exp')

def get_pois_overpass(area_name, city_name, poi_categories):
    """Get POIs using Overpass API for multiple categories within a 1km radius."""
    overpass_url = "https://overpass-api.de/api/interpreter"

    # --- 1. Define Search Area ---
    geocode_query = f"{area_name}, {city_name}"
    geocode_url = f"https://nominatim.openstreetmap.org/search?q={quote(geocode_query)}&format=json&limit=1"
    try:
        geocode_response = requests.get(geocode_url, headers={'User-Agent': 'MapMind/1.0'})
        geocode_response.raise_for_status()
        geocode_data = geocode_response.json()
    except requests.RequestException as e:
        print(f"Geocoding Error: {str(e)}")
        return {"error": f"Geocoding failed: {str(e)}"}

    if not geocode_data:
        return {"error": "Could not geocode area/city"}

    geocode_lat = float(geocode_data[0].get('lat'))
    geocode_lon = float(geocode_data[0].get('lon'))
    
    # Get bounding box for the area
    bbox = None
    if 'boundingbox' in geocode_data[0]:
        bbox_raw = geocode_data[0]['boundingbox']
        bbox = [float(bbox_raw[2]), float(bbox_raw[0]), float(bbox_raw[3]), float(bbox_raw[1])]
    else:
        radius_deg = 0.009
        bbox = [
            geocode_lon - radius_deg,
            geocode_lat - radius_deg,
            geocode_lon + radius_deg,
            geocode_lat + radius_deg
        ]

    # --- 2. Query for POIs by Category ---
    all_pois = {}
    
    # Expanded query to include more POI types
    overpass_query = f"""
    [out:json][timeout:300];
    (
        // Amenities
        nwr["amenity"](around:1500,{geocode_lat},{geocode_lon});

        // Leisure
        nwr["leisure"](around:1500,{geocode_lat},{geocode_lon});
        // Shops
        nwr["shop"](around:1500,{geocode_lat},{geocode_lon});
        // Offices
        nwr["office"](around:1500,{geocode_lat},{geocode_lon});
        // Public Transport
        nwr["public_transport"](around:1500,{geocode_lat},{geocode_lon});
        // Railways
        nwr["railway"~"^(station|halt|tram_stop)$"](around:1500,{geocode_lat},{geocode_lon});
        // Healthcare
        nwr["healthcare"](around:1500,{geocode_lat},{geocode_lon});
        // Education
        nwr["education"](around:1500,{geocode_lat},{geocode_lon});
    );
    out center;
    """
    
    try:
        response = requests.post(overpass_url, data=overpass_query)
        response.raise_for_status()
        data = response.json()
        
        # Process and categorize POIs
        pois = []
        for element in data.get('elements', []):
            tags = element.get('tags', {})
            
            # Get coordinates
            if element['type'] == 'node':
                lat, lon = element.get('lat'), element.get('lon')
            else:
                center = element.get('center', {})
                lat, lon = center.get('lat'), center.get('lon')
            
            if lat and lon:
                # Determine category based on tags
                category = None
                if 'amenity' in tags:
                    category = tags['amenity']
                elif 'shop' in tags:
                    category = f"shop_{tags['shop']}"
                elif 'leisure' in tags:
                    category = f"leisure_{tags['leisure']}"
                elif 'healthcare' in tags:
                    category = f"healthcare_{tags['healthcare']}"
                elif 'building' in tags:
                    category = f"building_{tags['building']}"
                elif 'office' in tags:
                    category = f"office_{tags['office']}"
                elif 'public_transport' in tags:
                    category = 'public_transport'
                elif 'railway' in tags:
                    category = f"railway_{tags['railway']}"
                
                if category:
                    if category not in all_pois:
                        all_pois[category] = []
                    
                    all_pois[category].append({
                        'lat': lat,
                        'lon': lon,
                        'tags': tags,
                        'type': element['type']
                    })
                
    except Exception as e:
        print(f"Error fetching POIs: {str(e)}")
    
    return {
        "pois": all_pois,
        "geocode": {
            "lat": geocode_lat,
            "lon": geocode_lon,
            "display_name": geocode_data[0].get('display_name', '')
        },
        "bbox": bbox
    }

def generate_complete_geojson(area_name, city_name, pois_data, bbox):
    """Generate a complete GeoJSON with boundary polygon and POIs using LLM."""
    
    # Create a prompt for the LLM to generate the GeoJSON
    prompt = f"""
    Create a complete GeoJSON FeatureCollection for {area_name}, {city_name} that includes:
    
    1. A boundary polygon feature with these properties:
       - type: "boundary"
       - name: "{area_name}, {city_name}"
       - fillColor: "#0070f3"
       - fillOpacity: 0.2
       - strokeColor: "#0070f3"
       - strokeWidth: 2
    
    2. Point features for each POI in this data:
    {json.dumps(pois_data, indent=2)}
    
    The boundary should be a simple polygon that covers the bounding box: {bbox}
    [west, south, east, north] = {bbox}
    
    Each POI should have these properties:
    - type: "poi"
    - category: (the POI category)
    - name: (the POI name if available, otherwise use the category)
    - color: (assign a unique color to each category)
    
    Return ONLY valid GeoJSON with no additional text or explanations.
    The response must be a complete, valid GeoJSON FeatureCollection.
    """
    
    try:
        response = model.generate_content(prompt)
        
        # Extract JSON from the response
        json_start = response.text.find('{')
        json_end = response.text.rfind('}') + 1
        
        if json_start == -1 or json_end == -1:
            # If no JSON found, create a basic GeoJSON with just the boundary
            return create_basic_geojson(bbox, area_name, city_name, pois_data)
            
        json_str = response.text[json_start:json_end]
        
        try:
            geojson = json.loads(json_str)
            # Validate basic GeoJSON structure
            if geojson.get('type') != 'FeatureCollection' or 'features' not in geojson:
                return create_basic_geojson(bbox, area_name, city_name, pois_data)
            return geojson
        except json.JSONDecodeError:
            return create_basic_geojson(bbox, area_name, city_name, pois_data)
            
    except Exception as e:
        print(f"Error generating GeoJSON: {str(e)}")
        return create_basic_geojson(bbox, area_name, city_name, pois_data)

# Add this function at the module level, before it's called
def generate_boundary_geojson(area_name, city_name, bbox):
    """Generate a simple GeoJSON with just the boundary polygon from bbox."""
    # Create boundary polygon from bbox
    boundary_coords = [
        [bbox[0], bbox[1]],  # Southwest
        [bbox[0], bbox[3]],  # Northwest
        [bbox[2], bbox[3]],  # Northeast
        [bbox[2], bbox[1]],  # Southeast
        [bbox[0], bbox[1]]   # Close the polygon
    ]
    
    # Create the GeoJSON structure with just the boundary
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [boundary_coords]
                },
                "properties": {
                    "type": "boundary",
                    "name": f"{area_name}, {city_name}",
                    "fillColor": "#0070f3",
                    "fillOpacity": 0.2,
                    "strokeColor": "#0070f3",
                    "strokeWidth": 2
                }
            }
        ]
    }
    
    return geojson

# Then fix the create_basic_geojson function to remove the nested definition
def create_basic_geojson(bbox, area_name, city_name, pois_data):
    """Create a basic GeoJSON with boundary and POIs as fallback."""
    # Create boundary polygon from bbox
    boundary_coords = [
        [bbox[0], bbox[1]],  # Southwest
        [bbox[0], bbox[3]],  # Northwest
        [bbox[2], bbox[3]],  # Northeast
        [bbox[2], bbox[1]],  # Southeast
        [bbox[0], bbox[1]]   # Close the polygon
    ]
    
    # Create the GeoJSON structure
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [boundary_coords]
                },
                "properties": {
                    "type": "boundary",
                    "name": f"{area_name}, {city_name}",
                    "fillColor": "#0070f3",
                    "fillOpacity": 0.2,
                    "strokeColor": "#0070f3",
                    "strokeWidth": 2
                }
            }
        ]
    }
    
    # Add POI features
    colors = ["#0088FE", "#00C49F", "#FFBB28", "#FF8042", "#AF19FF", "#FF1919"]
    color_index = 0
    
    for category, pois in pois_data.items():
        color = colors[color_index % len(colors)]
        color_index += 1
        
        for poi in pois:
            if poi.get('lat') and poi.get('lon'):
                geojson["features"].append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [poi['lon'], poi['lat']]
                    },
                    "properties": {
                        "type": "poi",
                        "category": category,
                        "name": poi.get('tags', {}).get('name', category),
                        "color": color
                    }
                })
    
    return geojson

def generate_pois_geojson(area_name, city_name, pois_data):
    """Generate a GeoJSON with just POI points."""
    geojson = {
        "type": "FeatureCollection",
        "features": []
    }
    
    # Define super-categories with colors - these will be used consistently
    super_categories = {
        "healthcare": {
            "color": "#0088FE",
            "display_name": "Healthcare",
            "patterns": ["hospital", "healthcare", "doctors", "pharmacy", "blood_bank", "optometrist", "alternative"]
        },
        "education": {
            "color": "#00C49F",
            "display_name": "Education",
            "patterns": ["school", "college", "university", "kindergarten", "training", "language_school", "education"]
        },
        "shopping": {
            "color": "#FFBB28",
            "display_name": "Shopping",
            "patterns": ["shop", "supermarket", "mall", "market", "bakery", "convenience"]
        },
        "food_drink": {
            "color": "#FF8042",
            "display_name": "Food & Drink",
            "patterns": ["restaurant", "cafe", "pub", "bar", "fast_food", "food_court", "ice_cream"]
        },
        "transport": {
            "color": "#AF19FF",
            "display_name": "Transport",
            "patterns": ["bus", "train", "station", "taxi", "parking", "transport"]
        },
        "financial": {
            "color": "#FF1919",
            "display_name": "Financial",
            "patterns": ["bank", "atm", "money", "financial", "insurance"]
        },
        "leisure": {
            "color": "#17BECF",
            "display_name": "Leisure",
            "patterns": ["leisure", "park", "garden", "playground", "swimming", "sports", "pitch", "track"]
        },
        "office": {
            "color": "#9467BD",
            "display_name": "Office",
            "patterns": ["office", "administrative", "government", "estate_agent", "tax", "telecommunication"]
        },
        "community": {
            "color": "#D62728",
            "display_name": "Community",
            "patterns": ["community", "social", "public", "toilets", "drinking_water", "bench", "library"]
        },
        "other": {
            "color": "#7F7F7F",
            "display_name": "Other",
            "patterns": []  # Catch-all for anything else
        }
    }
    
    # Create a dictionary to count POIs by super-category
    super_category_counts = {category: 0 for category in super_categories}
    
    # Process each POI and assign to a super-category
    for category, pois in pois_data.items():
        for poi in pois:
            if poi.get('lat') and poi.get('lon'):
                # Determine super-category
                super_category = "other"  # Default
                
                # Check each super-category's patterns
                for sc_name, sc_info in super_categories.items():
                    if any(pattern in category.lower() for pattern in sc_info["patterns"]):
                        super_category = sc_name
                        break
                
                # Increment count for this super-category
                super_category_counts[super_category] += 1
                
                # Add to GeoJSON with super-category color
                geojson["features"].append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [poi['lon'], poi['lat']]
                    },
                    "properties": {
                        "type": "poi",
                        "super_category": super_category,
                        "display_name": super_categories[super_category]["display_name"],
                        "category": category,
                        "name": poi.get('tags', {}).get('name', category),
                        "color": super_categories[super_category]["color"]
                    }
                })
    
    # Create pie chart data directly here to ensure color consistency
    pie_chart_data = []
    for sc_name, count in super_category_counts.items():
        if count > 0:  # Only include categories with POIs
            pie_chart_data.append({
                "name": super_categories[sc_name]["display_name"],
                "value": count,
                "color": super_categories[sc_name]["color"]
            })
    
    return geojson, super_categories, pie_chart_data

# Remove the global analysis_prompt and move it inside analyze_area function

@app.post("/analyze-area")
async def analyze_area(area_request: dict):
    city_name = area_request.get('city')
    area_name = area_request.get('area')
    full_area_name = f"{area_name}, {city_name}"

    print(f"\n=== Starting Analysis for: {full_area_name} ===")

    poi_categories = [
        "school", "hospital", "pharmacy", "supermarket", "grocery",
        "restaurant", "cafe", "bar", "pub", "bus_stop", "train_station",
        "park", "playground", "bank", "atm", "post_office"
    ]

    try:
        # Get POIs from Overpass API
        print("\n[1/3] Getting POIs from Overpass...")
        overpass_pois_data = get_pois_overpass(area_name, city_name, poi_categories)

        if "error" in overpass_pois_data:
            print(f"Error fetching POIs from Overpass: {overpass_pois_data['error']}")
            raise HTTPException(status_code=500, detail=overpass_pois_data['error'])

        categorized_pois = overpass_pois_data['pois']
        geocode = overpass_pois_data['geocode']
        bbox = overpass_pois_data.get('bbox')
        
        print(f"Total POIs after combining: {sum(len(pois) for pois in categorized_pois.values())}")
        
        # Generate GeoJSON with just POIs
        print("\n[3/3] Generating POIs GeoJSON...")
        pois_geojson, super_categories, pie_chart_data = generate_pois_geojson(area_name, city_name, categorized_pois)
        
        # Move the analysis prompt here
        analysis_prompt = f"""
        Imagine you're a local resident giving a friendly, conversational tour of {area_name}, {city_name}. 
        Create an engaging summary that covers:

        1. The neighborhood's vibe and lifestyle (based on the POIs)
        2. What makes this area special for a 15-minute city concept
        3. What daily life might look like here
        4. Any unique features or interesting combinations of amenities

        Use a conversational, first-person tone as if you're talking to a friend. 
        Include specific references to the POIs and their distribution:
        {json.dumps(categorized_pois, indent=2)}

        Make it personal and relatable, mentioning real scenarios like:
        - Morning coffee runs
        - Weekend activities
        - Daily conveniences
        - Community spots
    
        Keep it concise but engaging, around 3-4 sentences.
        
        Provide a structured JSON response with the following keys:
        - "summary": (in simple text, *NOT in JSON*) A concise summary of the living potential of the area, highlighting the strengths and weaknesses in each super-category.
        - "ai_rating": A numerical rating from 0 to 100 representing how well this area functions as a "15-minute city" where residents can access most daily needs within a 15-minute walk or bike ride.
        
        IMPORTANT: Only return the raw JSON, no additional text or explanations. The response must start with '{{' and end with '}}'.
        """

        # Get analysis
        try:
            response = model.generate_content(analysis_prompt)
            
            # Extract JSON from the response
            json_start = response.text.find('{')
            json_end = response.text.rfind('}') + 1
            
            if json_start == -1 or json_end == -1:
                raise ValueError("No JSON found in the response")
                
            json_str = response.text[json_start:json_end]
            analysis_results = json.loads(json_str)
            
            # Combine all results
            final_results = {
                "summary": analysis_results.get("summary", ""),
                "pie_chart_data": pie_chart_data,  # Use our pre-generated data instead of LLM's
                "ai_rating": analysis_results.get("ai_rating", 0),
                "geocode": geocode,
                "bbox": bbox,
                "geojson": pois_geojson,  # Include only the POIs GeoJSON
                "pois": categorized_pois  # Keep the POIs separate for the frontend
            }
            
            return final_results
            
        except Exception as e:
            print(f"LLM API Error: {str(e)}")
            raise HTTPException(status_code=502, detail=f"LLM API Error: {str(e)}")
    except Exception as e:
        print(f"Unexpected Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
'''HERE CHECKPOINT WORKING

uvicorn main:app --reload
npm start

'''
    
'''HERE CHECKPOINT WORKING

uvicorn main:app --reload
npm start
analysis_prompt = f"""
        You are a expert location analyser for people to move in there. Analyze the living potential of {full_area_name} based on these categorized Points of Interest (POIs):
        {json.dumps(categorized_pois, indent=2)}
        Focus on the following aspects which are important for living standards and dont be strict whatever data you get analyze that and provide your opnion and analysis based on that.
        
        Provide a structured JSON response with the following keys:
        - "summary":(in simple text) A concise summary of the living potential of the area,based on the POI of the data,you should consider the following factors like accessibility,amenities,transportation,security,healthcare,education,employment(and dont say like based on the data and expect other data, perform on this only)
        - "pie_chart_data": Data suitable for a pie chart visualizing the distribution of POI types. Include "name" and "value" for each category. Example: [{{"name": "Residential", "value": 30}}, {{"name": "Commercial", "value": 70}}]
        - "ai_rating": A numerical rating from 0 to 100 representing how well this area functions as a "15-minute city" where residents can access most daily needs within a 15-minute walk or bike ride.
        
        IMPORTANT: Only return the raw JSON, no additional text or explanations. The response must start with '{{' and end with '}}'.
        """
'''