# NPC Conversation Prompts
# =======================
# This file contains all the prompt templates used for NPC conversations and backstory generation.
# 
# MAIN PROMPTS IN USE:
# - get_conversation_prompt(): Used when player talks to NPCs during gameplay
# - get_backstory_prompt(): Used when generating NPC personalities/backstories
#
# EDITING INSTRUCTIONS:
# 1. Edit the main prompts above to change how NPCs behave
# 2. Variables like {npc_name} and {player_message} are automatically filled in by the code
#
# TIPS:
# - Keep conversation prompts concise for faster LLM responses
# - Use clear instructions about response length and style
# - Make sure to include the NPC's name throughout the prompt
# - Recent memories help NPCs remember what just happened

def get_conversation_prompt(npc_name, backstory_summary, memory_text, player_message):
    prompt = f"""
You are {npc_name}, an NPC in a 2D fantasy RPG.

Recent memories:
{memory_text}

Player just said:
{player_message}

Behavior:
- If the player's message is a greeting (hello/hi/hey/yo/greetings), reply with a short greeting.
- If the player's message asks about items, inventory, or anything you cannot verify from memories, reply: I don't know.
- Keep it natural and short.

Reply with only what you say to the player. One short, natural sentence on a single line. Do not include names, labels, examples, instructions, or quotes. Do not echo this prompt.
"""
    return prompt.strip()



def get_backstory_prompt(npc_name, location_description):
    """
    MAIN BACKSTORY PROMPT - Edit this to change how NPC personalities are generated
    
    Args:
        npc_name: The name of the NPC (e.g., "Bob")
        location_description: Description of the game world/location
    
    Returns:
        str: The complete prompt to send to the LLM for backstory generation
    """
    
    prompt = f"""Describe {npc_name}, a person living in {location_description}.

{npc_name} is"""
    
    return prompt 