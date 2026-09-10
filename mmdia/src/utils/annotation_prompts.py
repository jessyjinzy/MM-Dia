
PROMPT_SPEAKER_DIARIZATION = """{transcription}\n Given a video dialogue clip and the list of {count} raw English SRT lines from the movie "{movie}". 
Please perform speaker diarization. The output should be in JSON array where each element is a string: `"Speaker: English SRT line"`
Speaker: use **character names or identities** as speaker IDs. Identify characters only when absolutely certain, otherwise use distinct on-screen persona.
SRT line: the English SRT text (do not translate or alter raw content)

Please:
Identify distinct speakers as accurately as possible based on the visual and audio content.
Assign consistent speaker labels across the utterances.
Add fine-grained non-verbal sound annotations in <> (e.g., sigh, laughter, background sounds, ...) **within the line** where they occur.

OUTPUT FORMAT:
- If the clip audio represent a movie narrator's Voice-over or purely background music singing, output only: `"No conversation"`. Otherwise,
- *Directly* provide the final output in a JSON format containing the LIST of character annotations. Each entry should be a string, e.g.: 'Speaker1: Dialogue1 <non-verbal sound> Speaker2: Dialogue2'. No additional text or explanations. There are {count} SRT lines, make sure the output contains {count} entries with the i-th entry *STRICTLY* regarding to the i-th SRT. *No Overlapping between entries*.

Key Instructions:
If a single SRT line contains dialogue from multiple speakers, mark different speakers clearly within the entry for that SRT line.
If a complete sentence is split into multiple SRT lines, remain each SRT portion and add speaker multiple times.
If SRT line contains incorrect spellings (e.g., "l" -> "I", "lt's" -> "It's"), correct them in your output.
"""

PROMPT_ALIGNMENT = """
You are an expert in aligning the raw SRT with corresponding range of annotation. 
Your Goal: 
Given a list of raw SRT lines and a list of character annotations, adjust and re-align the corresponding annotations (with speakers and non-verbal annotations) into the range of each original SRT line.
Key Instructions:
Combine Annotations (N-to-1): If a single SRT line contains dialogue from multiple annotations, combine the relative portion into one entry for that SRT line. Mark different speakers clearly.
Split Annotations: If a single character annotation contains multiple SRT lines, split the annotation context precisely into each correct SRT portion. Add speaker to each corresponding SRT line.
Preserve All Details: Fully and precisely preserve the non-verbal sounds in angle brackets (e.g., <chewing>) attached to the lines in annotations.
Use Corrected Text: The character_annotation list contains corrected spellings (e.g., "l" -> "I", "lt's" -> "It's"). Use the corrected version in your output.
Output Format: *Directly* provide the final output in a JSON format containing the modified LIST of character annotations. Each entry should be a string, e.g.: 'Speaker1: Dialogue1 <non-verbal sound> Speaker2: Dialogue2'. No additional text or explanations.
There are {length} SRT lines, make sure the output contains {length} entries with the i-th entry *STRICTLY* regarding to the i-th SRT. *No Overlapping between entries*.
Else if the whole annotations are not related to the SRT lines, please directly return "No Annotation".

Now, process the following input:
Original SRT lines:
{original_srt_lines}

Character annotations:
{annotations}
"""

PROMPT_STYLE = """Here is a dialogue clip from the movie '{movie}', with characters {speaker} involved.\n
Your TASK: Speaking Style and Interaction Summary
You will analyze **how** characters speak and interact emotionally in this clip, based on the scenario and your knowledge of the characters.

Task 1: Tags
Generate three tags for the dialogue's expressive and interactional features, covering:
1. Relationship (e.g., \"couples argue\", \"workplace disputes\").
2. Interaction type (e.g., \"debate\", \"interruptions\", \"persuasion\").
3. Emotional tone or progression (e.g., \"growing tension\", \"suppressed anger\", \"fake politeness\").

Task 2: Speaking Style Summary
Write **one short, direct and natural sentence** that captures the speaking and emotional style of the dialogue:
- How characters speak, interact — in tone, rhythm, energy, or changes in emotional delivery (e.g., starts calm and grows hesitant, frequently interrupts, uses sarcasm).
- Use character names, relationships and other information if recognized.

Instructions:
- Focus only on **how** things are said, not the conversation content.
- Be specific, diverse, and **use simple, everyday language** — avoid vague or fancy phrasing. Use more verbs, less adjective. 
- Vary your expressions, avoid repeating words like \"escalate\", \"exaggerated\", \"speak\", \"exasperation\".
- Imagine you're telling someone how a scene *feels to hear* — not a formal summary.

Output in the JSON format:
{{
        \"Tags\": [\"list of tags\"],
        \"Summary\": \"the summary\"
}}
"""