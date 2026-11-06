"""Australian BOM weather assistant powered by LangChain and Groq."""

import argparse
import asyncio
import os
from dataclasses import asdict
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from bom_weather_api import BomClient, Coordinate
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_groq import ChatGroq


def get_coordinates(city_name: str):
    """
    Convert an Australian city or town name into latitude and longitude
    using the Open-Meteo Geocoding API.

    Args:
        city_name: City or town name, e.g. "Melbourne", "Horsham",
                   "Halls Gap", or "Stawell".

    Returns:
        tuple: (latitude, longitude, full_location_name)
        None: If the location cannot be found or an error occurs.
    """

    url = "https://geocoding-api.open-meteo.com/v1/search"

    params = {
        "name": city_name,
        "count": 10,
        "language": "en",
        "format": "json",
        "countryCode": "AU"
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=10
        )
        response.raise_for_status()

        data = response.json()
        results = data.get("results", [])

        if not results:
            print(f"Could not find Australian location: '{city_name}'")
            return None

        # Prefer an exact city/town name match
        search_name = city_name.strip().split(",")[0].casefold()

        location = next(
            (
                loc for loc in results
                if loc["name"].casefold() == search_name
            ),
            results[0]
        )

        lat = location["latitude"]
        lon = location["longitude"]

        name = location["name"]
        state = location.get("admin1", "")
        country = location.get("country", "Australia")

        full_name = ", ".join(
            part for part in [name, state, country] if part
        )

        return lat, lon, full_name

    except requests.exceptions.RequestException as e:
        print(f"Geocoding API error: {e}")
        return None

    except (ValueError, KeyError, TypeError) as e:
        print(f"Geocoding data error: {e}")
        return None


def format_bom_forecast(df):

    columns = {
        "date": "Date",
        "temp_max": "Max (°C)",
        "temp_min": "Min (°C)",
        "short_text": "Conditions",
        "rain_chance": "Rain Chance (%)",
        "rain_amount_min": "Rain Min (mm)",
        "rain_amount_max": "Rain Max (mm)",
        "uv_index": "UV Index",
        "uv_category": "UV Category",
        "fire_danger": "Fire Danger",
        "extended_text": "Forecast Details"
    }

    result = df[list(columns.keys())].copy()
    result = result.rename(columns=columns)

    # Format dates
    result["Date"] = pd.to_datetime(
        result["Date"]
    ).dt.strftime("%a, %d %b")

    # Format rainfall range
    result["Rainfall (mm)"] = (
        result["Rain Min (mm)"].fillna(0).map(lambda x: f"{x:g}")
        + " - "
        + result["Rain Max (mm)"].fillna(0).map(
            lambda x: f"{x:g}" if pd.notna(x) else "N/A"
        )
    )

    result = result.drop(
        columns=["Rain Min (mm)", "Rain Max (mm)"]
    )

    # Reorder columns
    result = result[
        [
            "Date",
            "Conditions",
            "Max (°C)",
            "Min (°C)",
            "Rain Chance (%)",
            "Rainfall (mm)",
            "UV Index",
            "UV Category",
            "Fire Danger",
            "Forecast Details"
        ]
    ]

    return result

@tool
async def get_weather_forecast(city_name: str) -> list[dict]:
    """
    Retrieve the 7-day daily weather forecast from the Australian
    Bureau of Meteorology (BOM) for a specified city or location.

    The function first converts the supplied city name into geographic
    coordinates (latitude and longitude) using the Open-Meteo Geocoding API.
    It then identifies the nearest available BOM forecast location to those
    coordinates and retrieves its daily weather forecast.

    Args:
        city_name (str):
            Name of the city, town, or location to retrieve the forecast for.
            For example: "Halls Gap", "Melbourne", "Stawell", or "Grampians".

    Returns:
        list[dict]:
            Formatted records containing up to 7 days of BOM daily forecast
            information, including:
            - date: Forecast date.
            - temp_max: Maximum expected temperature (°C).
            - temp_min: Minimum expected temperature (°C).
            - short_text: Short weather description.
            - extended_text: Detailed weather forecast.
            - rain_chance: Probability of rainfall (%).
            - rain_amount_min: Minimum expected rainfall (mm).
            - rain_amount_max: Maximum expected rainfall (mm).
            - uv_index: Forecast UV index.
            - uv_category: UV risk category.
            - fire_danger: Fire danger rating.

    Notes:
        - Weather forecast data comes from the Australian Bureau of
          Meteorology (BOM).
        - Open-Meteo is used only for geocoding, not weather forecasting.
        - The nearest BOM forecast town may differ from the requested city.
        - Forecast accuracy depends on the distance between the requested
          location and the nearest BOM forecast location.

    Example:
        get_weather_forecast("Halls Gap")
    """



    geo_data = get_coordinates(city_name)
    if not geo_data:
        return [{"error": f"Location not found: {city_name}"}]
    lat, long, loc = geo_data
    location = Coordinate(
        latitude=lat,
        longitude=long
    )
    print (f"Fetching weather forecast for {loc} at coordinates: {lat}, {long}")

    async with BomClient() as client:

        # Find nearest BOM forecast town
        towns = await client.get_forecast_locations()
        town, distance = BomClient.nearest(
            location, towns, limit=1
        )[0]

        print(f"Forecast location: {town.name}")
        print(f"Distance: {distance:.1f} km")

        # Get BOM daily forecast
        forecast = await client.get_daily_forecast(town)

        # Extract up to 7 days
        records = [
            asdict(day)
            for day in forecast.days[:7]
        ]

        df = pd.json_normalize(records)
        formatted_df = format_bom_forecast(df)
        formatted_df = formatted_df.astype(object).where(pd.notna(formatted_df), None)
        return formatted_df.to_dict(orient="records")

@tool
async def get_hourly_forecast(
    city_name: str,
    forecast_date: str | None = None,
    start_hour: int | None = None,
    end_hour: int | None = None,
    weather_fields: list[str] | None = None
) -> list[dict]:
    """
Retrieve hourly weather forecasts for an Australian location
using the Bureau of Meteorology (BOM).

Use this tool when the user asks about:
- Weather at a specific hour or time of day.
- Wind speed, gusts, or direction.
- Hourly temperature, humidity, or dew point.
- Rainfall probability or expected rainfall.
- Weather changes throughout a particular day.

Tool Usage Rules:
- Always specify forecast_date when the user requests a date.
- For full-day requests, specify forecast_date and leave
  start_hour and end_hour as None.
- If the user specifies a time of day, restrict the request
  to that period:
    Morning: 06:00-12:00
    Afternoon: 12:00-18:00
    Evening: 18:00-24:00
- For overnight periods crossing midnight, make separate
  calls for each date.
- Request only the weather_fields relevant to the question.
- For wind questions, request wind_speed, wind_gust,
  and wind_direction.
- For temperature questions, request temperature and
  optionally apparent_temperature.
- For rain questions, request rain_chance, rain_amount_min,
  and rain_amount_max.
- For humidity questions, request humidity and optionally dew_point.
- Do not request unnecessary weather variables.
- Do not assume hourly data is available for every future date.

Args:
    city_name:
        Australian city or town, e.g. "Horsham" or "Melbourne".

    forecast_date:
        Optional local forecast date in YYYY-MM-DD format.
        Resolve relative dates such as "tomorrow" before calling.
        If omitted, use the earliest available forecast date.

    start_hour:
        Optional starting hour (0-23), inclusive.
        For example, 12 means noon.

    end_hour:
        Optional ending hour (1-24), exclusive.
        For example, 18 means before 6 PM.

    weather_fields:
        Optional list of weather variables to retrieve.
        Available fields:
        temperature, apparent_temperature, dew_point,
        humidity, wind_speed, wind_gust, wind_direction,
        rain_chance, rain_amount_min, rain_amount_max,
        uv_index, icon_descriptor, is_night.

        Examples:
        ["wind_speed", "wind_gust", "wind_direction"]
        ["temperature", "humidity"]

        If omitted, return a compact set of common variables.

Returns:
    List of hourly forecast records filtered by the requested
    date, time range, and weather variables.

    Timestamps are converted to Australia/Melbourne timezone.
    An empty list means no forecasts matched the filters.
"""

    # 1. Validate parameters
    if start_hour is not None and not 0 <= start_hour <= 23:
        raise ValueError("start_hour must be between 0 and 23")

    if end_hour is not None and not 1 <= end_hour <= 24:
        raise ValueError("end_hour must be between 1 and 24")

    if (
        start_hour is not None
        and end_hour is not None
        and start_hour >= end_hour
    ):
        raise ValueError("start_hour must be less than end_hour")

    if forecast_date is not None:
        datetime.strptime(forecast_date, "%Y-%m-%d")

    # 2. Resolve coordinates
    geo_data = get_coordinates(city_name)

    if not geo_data:
        return [{"error": f"Location not found: {city_name}"}]

    lat, lon, loc = geo_data
    location = Coordinate(latitude=lat, longitude=lon)

    # For Victorian locations. Replace with a location-aware
    # timezone resolver if supporting all Australian states.
    timezone = ZoneInfo("Australia/Melbourne")

    # 3. Retrieve BOM hourly forecast
    async with BomClient() as client:
        forecast = await client.get_hourly_forecast(location)

    records = [asdict(hour) for hour in forecast.hours]

    if not records:
        return []

    df = pd.json_normalize(records)

    # 4. Convert timestamps to local time
    df["time"] = (
        pd.to_datetime(df["time"], utc=True)
        .dt.tz_convert(timezone)
    )

    # 5. Filter by date
    if forecast_date is not None:
        df = df[
            df["time"].dt.strftime("%Y-%m-%d") == forecast_date
        ]
    else:
        # Default to earliest available local forecast date
        earliest_date = df["time"].dt.date.min()
        df = df[df["time"].dt.date == earliest_date]

    # 6. Filter by hour
    if start_hour is not None:
        df = df[df["time"].dt.hour >= start_hour]

    if end_hour is not None:
        df = df[df["time"].dt.hour < end_hour]

    # 7. Select weather fields
    default_fields = [
        "temperature",
        "wind_speed",
        "wind_gust",
        "rain_chance"
    ]

    selected_fields = (
        weather_fields if weather_fields is not None
        else default_fields
    )

    available_fields = [
        "temperature",
        "apparent_temperature",
        "dew_point",
        "humidity",
        "wind_speed",
        "wind_gust",
        "wind_direction",
        "rain_chance",
        "rain_amount_min",
        "rain_amount_max",
        "uv_index",
        "icon_descriptor",
        "is_night"
    ]

    invalid_fields = set(selected_fields) - set(available_fields)

    if invalid_fields:
        raise ValueError(
            f"Invalid weather fields: {sorted(invalid_fields)}"
        )

    columns = ["time"] + [
        field for field in selected_fields
        if field in df.columns
    ]

    df = df[columns].copy()

    # 8. Format time and missing values
    df["time"] = df["time"].dt.strftime(
        "%Y-%m-%d %H:%M %Z"
    )

    df = df.astype(object).where(pd.notna(df), None)

    return df.to_dict(orient="records")

SYSTEM_PROMPT = """
  You are an intelligent weather assistant specializing in
  Australian weather forecasts from the Bureau of Meteorology (BOM).

  Your responsibility is to provide accurate, clear, and useful
  weather information for Australian cities, towns, and locations.

  AVAILABLE TOOLS:

  1. get_weather_forecast
    - Retrieves daily weather forecasts for up to 7 days.
    - Use for daily forecasts, multi-day comparisons, rainfall,
      UV index, fire danger, and outdoor activity planning.

  2. get_hourly_forecast
    - Retrieves hourly weather forecasts.
    - Use for questions about specific times, wind, humidity,
      dew point, temperature, and hourly rainfall.
    - Follow the tool's description to select appropriate
      dates, time ranges, and weather variables.
    - Minimize unnecessary hourly data retrieval.

  TOOL SELECTION:
  - Always use the appropriate forecasting tool when answering
    questions about weather conditions or forecasts.
  - Use get_weather_forecast for daily or multi-day forecasts.
  - Use get_hourly_forecast for hourly forecasts or variables
    requiring hourly data.
  - Use both tools when the question requires daily and hourly data.
  - Do not call tools unnecessarily.
  - Never invent weather values or assume weather conditions.

  FORECAST INTERPRETATION:
  - Use only the weather information returned by the tools.
  - Distinguish rainfall probability (%) from rainfall amount (mm).
  - Consider temperature, rain, wind, UV, and fire danger when
    recommending outdoor activities.
  - Compare weather conditions across days when relevant.
  - Mention the forecast location when available in tool results.
  - If the forecast is based on a nearby BOM location, explain
    this when the tool provides that information.
  - If requested data is unavailable, clearly explain the limitation.
  - Do not present forecast values as real-time observations.

  RESPONSE GUIDELINES:
  - Use Australian units:
      Temperature: Celsius (°C)
      Wind: kilometres per hour (km/h)
      Rainfall: millimetres (mm)
  - Provide clear, concise, and user-friendly responses.
  - Summarize important conditions rather than listing every
    hourly observation unless the user requests detailed data.
  - Highlight relevant weather risks for outdoor activities.
  - Use tables for multi-day comparisons when helpful.
  - Do not include weather variables unrelated to the question.

  LIMITATIONS:
  - The available tools provide weather forecasts, not live
    weather observations.
  - Do not claim to provide current observations or air quality
    index (AQI) without an appropriate tool.
  - Hourly forecast coverage may be shorter than daily coverage.

  Your goal is to help users understand upcoming weather
  conditions and make informed decisions about outdoor
  rock climbing.
  """


def create_weather_agent(api_key: str | None = None):
    """Create a new LangChain agent with the current Melbourne date context."""
    load_dotenv()
    key = api_key or os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is missing. Configure it in environment or Streamlit secrets.")
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, api_key=key)
    current_date = datetime.now(ZoneInfo("Australia/Melbourne")).strftime("%A, %Y-%m-%d")
    return create_agent(
        model=llm,
        tools=[get_weather_forecast, get_hourly_forecast],
        system_prompt=SYSTEM_PROMPT + (
            f"\nCurrent date in Melbourne: {current_date}. "
            "Use this date to interpret relative dates such as today and tomorrow."
        ),
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the Australian BOM weather agent a question.")
    parser.add_argument("question", nargs="*", help="Weather question (if omitted, prompt for one).")
    args = parser.parse_args()
    question = " ".join(args.question).strip() or input("Weather question: ").strip()
    if not question:
        parser.error("Please provide a weather question.")
    agent = create_weather_agent()
    result = await agent.ainvoke({"messages": [("user", question)]})
    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
