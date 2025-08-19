# agent/schemas.py
SCHEMA_WEEKLY_PLAN = r"""
{
  "type": "object",
  "properties": {
    "week": {"type":"integer"},
    "strategy_summary": {"type":"string"},
    "start_sit": {
      "type":"object",
      "properties": {
        "moves": {
          "type":"array",
          "items": {
            "type":"object",
            "properties": {
              "player_id":{"type":"string"},
              "player_name":{"type":"string"},
              "from":{"type":"string"},
              "to":{"type":"string"},
              "reason":{"type":"string"}
            },
            "required":["player_id","to"]
          }
        },
        "notes":{"type":"string"}
      },
      "required":["moves"]
    },
    "waivers": {
      "type":"array",
      "items":{
        "type":"object",
        "properties": {
          "rank":{"type":"integer"},
          "player_id":{"type":"string"},
          "player_name":{"type":"string"},
          "bid":{"type":"integer"},
          "drop_player_id":{"type":"string"},
          "epar":{"type":["number","string"]},
          "reason":{"type":"string"}
        },
        "required":["rank","player_id","bid"]
      }
    },
    "trades": {
      "type":"array",
      "items":{
        "type":"object",
        "properties": {
          "to_team_key":{"type":"string"},
          "give_ids":{"type":"array","items":{"type":"string"}},
          "get_ids":{"type":"array","items":{"type":"string"}},
          "msg":{"type":"string"},
          "ev_median":{"type":"number"},
          "ev_worst":{"type":"number"},
          "justification":{"type":"string"}
        },
        "required":["to_team_key","give_ids","get_ids"]
      }
    },
    "blocks": {
      "type":"array",
      "items":{
        "type":"object",
        "properties": {
          "player_id":{"type":"string"},
          "player_name":{"type":"string"},
          "bid":{"type":"integer"},
          "reason":{"type":"string"}
        },
        "required":["player_id","bid"]
      }
    },
    "contingencies":{
      "type":"array",
      "items":{
        "type":"object",
        "properties":{
          "if":{"type":"string"},
          "then":{"type":"string"},
          "reason":{"type":"string"}
        },
        "required":["if","then"]
      }
    },
    "policy_checks":{"type":"array","items":{"type":"string"}},
    "risks":{"type":"array","items":{"type":"string"}}
  },
  "required":["week","strategy_summary","start_sit","waivers"]
}
"""

SCHEMA_START_SIT = r"""
{
  "type":"object",
  "properties": {
    "moves": {
      "type":"array",
      "items":{
        "type":"object",
        "properties": {
          "player_id":{"type":"string"},
          "player_name":{"type":"string"},
          "from":{"type":"string"},
          "to":{"type":"string"},
          "reason":{"type":"string"}
        },
        "required":["player_id","to"]
      }
    },
    "notes":{"type":"string"}
  },
  "required":["moves"]
}
"""

SCHEMA_WAIVERS = r"""
{
  "type":"object",
  "properties": {
    "claims": {
      "type":"array",
      "items": {
        "type":"object",
        "properties":{
          "rank":{"type":"integer"},
          "player_id":{"type":"string"},
          "player_name":{"type":"string"},
          "bid":{"type":"integer"},
          "drop_player_id":{"type":"string"},
          "epar":{"type":["number","string"]},
          "reason":{"type":"string"}
        },
        "required":["rank","player_id","bid"]
      }
    },
    "notes":{"type":"string"}
  },
  "required":["claims"]
}
"""

SCHEMA_TRADES = r"""
{
  "type":"object",
  "properties": {
    "offers": {
      "type":"array",
      "items":{
        "type":"object",
        "properties": {
          "to_team_key":{"type":"string"},
          "give_ids":{"type":"array","items":{"type":"string"}},
          "get_ids":{"type":"array","items":{"type":"string"}},
          "msg":{"type":"string"},
          "ev_median":{"type":"number"},
          "ev_worst":{"type":"number"},
          "justification":{"type":"string"}
        },
        "required":["to_team_key","give_ids","get_ids"]
      }
    }
  },
  "required":["offers"]
}
"""
