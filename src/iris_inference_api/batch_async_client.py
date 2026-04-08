import asyncio

import httpx


async def predict(client, features):
    try:
        r = await client.post(
            "http://127.0.0.1:8000/predict",
            json={"features": features},
            timeout=5.0
        )

        # Raises exception if HTTP status is 4xx or 5xx
        r.raise_for_status()

        return r.json()

    except httpx.HTTPStatusError as e:
        print(f"HTTP error for features {features}: {e.response.status_code}")
        return None

    except httpx.RequestError as e:
        print(f"Network error for features {features}: {e}")
        return None

    except Exception as e:
        print(f"Unexpected error for features {features}: {e}")
        return None


async def main():
    try:
        async with httpx.AsyncClient() as client:
            tasks = [
                predict(client, [6.1, 2.8, 4.7, 1.2]),
                predict(client, [5.7, 3.8, 1.7, 0.3]),
                predict(client, [7.7, 2.6, 6.9, 2.3]),
            ]

            # correct predictions are [1, 0, 2]

            results = await asyncio.gather(*tasks)

            print(results)

    except Exception as e:
        print(f"Main failed: {e}")


asyncio.run(main())