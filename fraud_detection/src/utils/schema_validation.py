from jsonschema import validate, Draft7Validator
import json

tx_schema = {
  "type": "object",
  "properties": {
    "tx_id": {"type": "string"},
    "user_id": {"type": "number"},
    "amount": {"type": "number"},
    "device": {"type": "string"},
    "ip_hash": {"type": "number"},
    "ts": {"type": "string", "format":"date-time"},
    "metadata": {"type":"object"},
  },
  "required": ["tx_id","user_id","amount","device","ip_hash","ts"]
}

def validate_tx(tx):
    errors = list(Draft7Validator(tx_schema).iter_errors(tx))
    if errors:
        raise ValueError(errors[0].message)
