import json
from datetime import datetime
from zoneinfo import ZoneInfo  # Python 3.9+

def lambda_handler(event, context):
    now_pst = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S %Z")

    return {
        "messageVersion": "1.0",
        "response": {
            "actionGroup": event.get("actionGroup"),
            "apiPath": event.get("apiPath"),
            "httpMethod": event.get("httpMethod"),
            "httpStatusCode": 200,
            "responseBody": {
                "application/json": {
                    "body": json.dumps({
                        "current_time_pst": now_pst,
                        "note": "Time is returned in Pacific Time (PST/PDT)."
                    })
                }
            }
        }
    }
