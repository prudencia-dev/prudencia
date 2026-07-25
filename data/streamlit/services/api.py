import requests


API_URL = "http://api:8000"


class PrudenciaAPI:

    @staticmethod
    def _parse_response(response: requests.Response):
        try:
            data = response.json()
        except ValueError:
            data = response.text

        if not response.ok:
            raise RuntimeError(
                f"Erreur API {response.status_code} : {data}"
            )

        return data

    @staticmethod
    def get(endpoint: str):
        response = requests.get(
            f"{API_URL}{endpoint}",
            timeout=60,
        )

        return PrudenciaAPI._parse_response(response)

    @staticmethod
    def post(
        endpoint: str,
        json=None,
        files=None,
        data=None,
    ):
        response = requests.post(
            f"{API_URL}{endpoint}",
            json=json,
            files=files,
            data=data,
            timeout=600,
        )

        return PrudenciaAPI._parse_response(response)

    @staticmethod
    def delete(endpoint: str):
        response = requests.delete(
            f"{API_URL}{endpoint}",
            timeout=60,
        )

        return PrudenciaAPI._parse_response(response)
    
    @staticmethod
    def delete(
        endpoint: str,
    ) -> dict:

        response = requests.delete(
            f"{API_URL}{endpoint}",
            timeout=300,
        )

        response.raise_for_status()

        return response.json()