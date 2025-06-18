from pydantic import BaseModel, Field
import requests
from mcp.server.fastmcp import FastMCP
import os
import statistics

mcp = FastMCP("solar_server")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# SOLAR_API_KEY = os.getenv("SOLAR_API_KEY")

# --- Tool Schemas ---
class GeocodeParams(BaseModel):
    address: str = Field(..., description="Residential address to geocode")

class SolarInsightsParams(BaseModel):
    latitude: float = Field(..., description="Latitude coordinate")
    longitude: float = Field(..., description="Longitude coordinate")

# --- Tool Implementations ---
@mcp.tool()
def geocode(params: GeocodeParams) -> dict:
    """Convert address to geographic coordinates"""
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    print("Params: ", params.address)

    response = requests.get(url, params={
        "address": params.address,
        "key": GOOGLE_API_KEY
    })

    print("Response: ", response)
    
    if response.status_code != 200:
        return {"error": "Geocoding API error"}
    
    data = response.json()

    print("Data: ", data)
    
    if not data.get("results"):
        return {"error": "No results found"}
    
    location = data["results"][0]["geometry"]["location"]
    return {
        "output": {
            "latitude": location["lat"],
            "longitude": location["lng"],
            "formatted_address": data["results"][0]["formatted_address"]
        }
    }

@mcp.tool()
def solar_insights(params: SolarInsightsParams) -> dict:
    """Get solar potential insights for location"""
    url = "https://solar.googleapis.com/v1/buildingInsights:findClosest"
    print("inside solar insights")
    print("Params: ", params.latitude, params.longitude)
    
    response = requests.get(url, params={
        "location.latitude": params.latitude,
        "location.longitude": params.longitude,
        "requiredQuality": "HIGH",
        "key": GOOGLE_API_KEY
    })
    
    if response.status_code != 200:
        return {"error": "Solar API error"}
    
    data = response.json()
    # print("Data: ", data)
    return process_solar_data(data)

def process_solar_data(data: dict) -> dict:
    """Process solar API response into structured format with additional elements"""
    sp = data.get("solarPotential", {})
    
    # Extract median panel config
    panel_configs = sp.get("solarPanelConfigs", [])
    median_panels = None
    median_energy = None
    if panel_configs:
        panels_counts = [cfg["panelsCount"] for cfg in panel_configs]
        median_panels = int(statistics.median(panels_counts))
        median_config = next((cfg for cfg in panel_configs 
                            if cfg["panelsCount"] == median_panels), None)
        if median_config:
            median_energy = median_config["yearlyEnergyDcKwh"]

    # Find ideal south-facing segment (azimuth closest to 180°)
    roof_segments = sp.get("roofSegmentStats", [])
    ideal_segment = None
    min_azimuth_diff = 360
    max_exposure = 0
    for rs in roof_segments:
        azimuth = rs.get("azimuthDegrees", 0)
        diff = abs(azimuth - 180)
        if diff > 180:
            diff = 360 - diff
        if diff < min_azimuth_diff:
            min_azimuth_diff = diff
            ideal_segment = rs
        
        # Track max solar exposure
        exposure = rs.get("stats", {}).get("sunshineQuantiles", [0])[-1]
        if exposure > max_exposure:
            max_exposure = exposure

    # Extract financial benefits
    financial_analyses = []
    for fa in sp.get("financialAnalyses", []):
        cash_savings = fa.get("cashPurchaseSavings", {})
        financial_details = fa.get("financialDetails", {})
        financial_analyses.append({
            "monthly_bill": fa.get("monthlyBill", {}).get("units"),
            "savings_20yr": cash_savings.get("savings", {}).get("savingsLifetime", {}).get("units"),
            "payback_years": cash_savings.get("paybackYears"),
            "federal_incentive": financial_details.get("federalIncentive", {}).get("units"),
            "state_incentive": financial_details.get("stateIncentive", {}).get("units"),
            "utility_incentive": financial_details.get("utilityIncentive", {}).get("units"),
            "net_cost": cash_savings.get("outOfPocketCost", {}).get("units")
        })

    # Calculate annual sunlight hours (median from wholeRoofStats)
    sunshine_quantiles = sp.get("wholeRoofStats", {}).get("sunshineQuantiles", [])
    median_sunlight = sunshine_quantiles[5] if len(sunshine_quantiles) > 5 else None

    # Calculate carbon offset
    max_config = next((c for c in panel_configs 
                      if c["panelsCount"] == sp.get("maxArrayPanelsCount")), None)
    carbon_offset = None
    if max_config and sp.get("carbonOffsetFactorKgPerMwh"):
        carbon_offset = (max_config["yearlyEnergyDcKwh"] / 1000) * sp["carbonOffsetFactorKgPerMwh"]

    # Classify orientation quality
    def classify_orientation(azimuth):
        if 135 <= azimuth <= 225: return "Excellent (South-facing)"
        elif 90 <= azimuth < 135 or 225 < azimuth <= 270: return "Good (SE/SW)"
        else: return "Fair"

    return {
        "solar_potential": {
            "max_panels": sp.get("maxArrayPanelsCount"),
            "panel_dimensions": {
                "height": sp.get("panelHeightMeters"),
                "width": sp.get("panelWidthMeters")
            },
            "system_lifespan": sp.get("panelLifetimeYears"),
            "roof_area": sp.get("wholeRoofStats", {}).get("areaMeters2"),
            "annual_sunlight_hours": median_sunlight,
            "median_panels_config": {
                "panels": median_panels,
                "annual_energy_kwh": median_energy
            },
            "energy_profiles": [
                {"panels": cfg["panelsCount"], "energy_kwh": cfg["yearlyEnergyDcKwh"]} 
                for cfg in panel_configs
            ],
            "annual_carbon_offset_kg": carbon_offset
        },
        "roof_analysis": [{
            "pitch_deg": rs["pitchDegrees"],
            "azimuth_deg": rs["azimuthDegrees"],
            "solar_exposure": rs.get("stats", {}).get("sunshineQuantiles", [0])[-1],
            "orientation_quality": classify_orientation(rs.get("azimuthDegrees", 0))
        } for rs in roof_segments],
        "financial_analysis": financial_analyses,
        "ideal_placement": {
            "recommended_azimuth": ideal_segment.get("azimuthDegrees") if ideal_segment else None,
            "pitch": ideal_segment.get("pitchDegrees") if ideal_segment else None,
            "max_exposure_segment": max_exposure
        } if ideal_segment else None
    }


def main():
    #uv --port 8000 run my_mcp_server
    mcp.run(transport="sse")

if __name__ == "__main__":
    main()
