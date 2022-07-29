import os
from fastapi.testclient import TestClient
from .app.server import app

os.chdir("app")
client = TestClient(app)

""" 
This part is optional. 

We've built our web application, and containerized it with Docker.
But imagine a team of ML engineers and scientists that needs to maintain, improve and scale this service over time. 
It would be nice to write some tests to ensure we don't regress! 

  1. `Pytest` is a popular testing framework for Python. If you haven't used it before, take a look at https://docs.pytest.org/en/7.1.x/getting-started.html to get started and familiarize yourself with this library.
   
  2. How do we test FastAPI applications with Pytest? Glad you asked, here's two resources to help you get started:
    (i) Introduction to testing FastAPI: https://fastapi.tiangolo.com/tutorial/testing/
    (ii) Testing FastAPI with startup and shutdown events: https://fastapi.tiangolo.com/advanced/testing-events/

"""


def test_root():
    """
    Test the root ("/") endpoint, which just returns a {"Hello": "World"} json response
    When you need your event handlers (startup and shutdown) to run in your tests, you can use the TestClient with a with statement
    More info:https://fastapi.tiangolo.com/advanced/testing-events/
    """
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"Hello": "World"}


def test_predict_empty():
    """
    Test the "/predict" endpoint, with an empty request body
    When you need your event handlers (startup and shutdown) to run in your tests, you can use the TestClient with a with statement
    More info:https://fastapi.tiangolo.com/advanced/testing-events/
    """
    with TestClient(app) as client:
        response = client.post("/predict", json={})
        assert response.status_code == 422
        response_body = {
            "detail": [
                {"loc": ["body", "source"], "msg": "field required", "type": "value_error.missing"},
                {"loc": ["body", "url"], "msg": "field required", "type": "value_error.missing"},
                {"loc": ["body", "title"], "msg": "field required", "type": "value_error.missing"},
                {
                    "loc": ["body", "description"],
                    "msg": "field required",
                    "type": "value_error.missing",
                },
            ]
        }
        assert response.json() == response_body


def test_predict_en_lang():
    """
    Test the "/predict" endpoint, with an input text in English (you can use one of the test cases provided in README.md)
    When you need your event handlers (startup and shutdown) to run in your tests, you can use the TestClient with a with statement
    More info:https://fastapi.tiangolo.com/advanced/testing-events/
    """
    with TestClient(app) as client:
        test_case_1 = {
            "source": "BBC Technology",
            "url": "http://news.bbc.co.uk/go/click/rss/0.91/public/-/2/hi/business/4144939.stm",
            "title": "System gremlins resolved at HSBC",
            "description": "Computer glitches which led to chaos for HSBC customers on Monday are fixed, the High Street bank confirms.",
        }

        response = client.post("/predict", json=test_case_1)
        assert response.status_code == 200
        assert response.json().get("label") == "Sci/Tech"

        # Note -> we avoid using an assert on response body score (as shown below) cuz of the randomness of decimal scores
        # assert response.json() == {
        #                             "scores": {
        #                                 "Business": 0.16220231669700727,
        #                                 "Entertainment": 0.32445776177698776,
        #                                 "Health": 0.04300249699121628,
        #                                 "Music Feeds": 0.008064347478608194,
        #                                 "Sci/Tech": 0.2791197520153585,
        #                                 "Software and Developement": 0.023869893828788336,
        #                                 "Sports": 0.13936834978605475,
        #                                 "Toons": 0.019915081425978923
        #                             },
        #                             "label": "Entertainment"
        #                             }

        test_case_2 = {
            "source": "Yahoo World",
            "url": "http://us.rd.yahoo.com/dailynews/rss/world/*http://story.news.yahoo.com/news?tmpl=story2u=/nm/20050104/bs_nm/markets_stocks_us_europe_dc",
            "title": "Wall Street Set to Open Firmer (Reuters)",
            "description": "Reuters - Wall Street was set to start higher on\Tuesday to recoup some of the prior session's losses, though high-profile retailer Amazon.com  may come under\pressure after a broker downgrade.",
        }

        response = client.post("/predict", json=test_case_2)
        # response code could also be 422 for -> Error: Unprocessable Entity
        assert response.status_code == 200

        test_case_3 = {
            "source": "New York Times",
            "url": "",
            "title": "Weis chooses not to make pickoff",
            "description": "Bill Belichick won't have to worry about Charlie Weis raiding his coaching staff for Notre Dame. But we'll have to see whether new Miami Dolphins coach Nick Saban has an eye on any of his former assistants.",
        }

        response = client.post("/predict", json=test_case_3)
        assert response.status_code == 200
        assert response.json().get("label") == "Entertainment"


def test_predict_es_lang():
    """
    Test the "/predict" endpoint, with an input text in Spanish.
    Does the tokenizer and classifier handle this case correctly? Does it return an error?
    When you need your event handlers (startup and shutdown) to run in your tests, you can use the TestClient with a with statement
    More info:https://fastapi.tiangolo.com/advanced/testing-events/
    """
    with TestClient(app) as client:
        test_case_spanish = {
            "source": "El Mundo",
            "url": "http://www.elmundo.es/ciencia/ciencia-y-salud/2016-01-04/un-cientifico-descubrio-que-el-hombre-no-es-un-bebe.html",
            "title": "Un científico descubrió que el hombre no es un bebé",
            "description": "El científico de la Universidad de Stanford, Eric W. Weis, ha descubierto que el hombre no es un bebé.",
        }

        response = client.post("/predict", json=test_case_spanish)
        assert response.status_code == 200
        assert response.json().get("label") == "Sci/Tech"


def test_predict_non_ascii():
    """
    Test the "/predict" endpoint, with an input text that has non-ASCII characters.
    Does the tokenizer and classifier handle this case correctly? Does it return an error?
    When you need your event handlers (startup and shutdown) to run in your tests, you can use the TestClient with a with statement
    More info:https://fastapi.tiangolo.com/advanced/testing-events/
    """
    with TestClient(app) as client:
        test_case_non_ascii = {
            "source": "New York Times",
            "url": "",
            "title": "Weis chooses not to make pickoff",
            "description": "This is a text with non-ASCII characters 日本人 中國的 ~=[]()%+@;’#!$_&- éè ;∞¥₤€. We hopè you find it inform@tiv€",
        }

        response = client.post("/predict", json=test_case_non_ascii)
        assert response.status_code == 200
        assert response.json().get("label") == "Entertainment"
