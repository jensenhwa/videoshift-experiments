"""
Collections of fixed prompts that classifiers can use.
Some are collections of the best-performing single prompts we found,
others are from the original CLIP paper or the TIP-Adapter paper.
The None key is used for the no-prompt scenario.
"""
PROMPT_ENSEMBLES = {
    None: ["{}"],
    "simple_photo": ["a photo of {}"],
    "simple_video": ["a video of {}"],
    "vid_action": [
        "i am {}",
        "the video shows me {}",
        "a photo showing a {}",
        "a photo showing the activity of {}"
    ],
    "tip_adapter": [
        "itap of a {}",
        "a bad photo of the {}",
        "a origami {}",
        "a photo of the large {}",
        "a {} in a video game",
        "art of the {}",
        "a photo of the small {}"
    ],
    "clip_kinetics": [
        'a photo of {}.',
        'a photo of a person {}.',
        'a photo of a person using {}.',
        'a photo of a person doing {}.',
        'a photo of a person during {}.',
        'a photo of a person performing {}.',
        'a photo of a person practicing {}.',
        'a video of {}.',
        'a video of a person {}.',
        'a video of a person using {}.',
        'a video of a person doing {}.',
        'a video of a person during {}.',
        'a video of a person performing {}.',
        'a video of a person practicing {}.',
        'a example of {}.',
        'a example of a person {}.',
        'a example of a person using {}.',
        'a example of a person doing {}.',
        'a example of a person during {}.',
        'a example of a person performing {}.',
        'a example of a person practicing {}.',
        'a demonstration of {}.',
        'a demonstration of a person {}.',
        'a demonstration of a person using {}.',
        'a demonstration of a person doing {}.',
        'a demonstration of a person during {}.',
        'a demonstration of a person performing {}.',
        'a demonstration of a person practicing {}.',
    ],
    "clip_ucf": [
        'a photo of a person {}.',
        'a video of a person {}.',
        'a example of a person {}.',
        'a demonstration of a person {}.',
        'a photo of the person {}.',
        'a video of the person {}.',
        'a example of the person {}.',
        'a demonstration of the person {}.',
        'a photo of a person using {}.',
        'a video of a person using {}.',
        'a example of a person using {}.',
        'a demonstration of a person using {}.',
        'a photo of the person using {}.',
        'a video of the person using {}.',
        'a example of the person using {}.',
        'a demonstration of the person using {}.',
        'a photo of a person doing {}.',
        'a video of a person doing {}.',
        'a example of a person doing {}.',
        'a demonstration of a person doing {}.',
        'a photo of the person doing {}.',
        'a video of the person doing {}.',
        'a example of the person doing {}.',
        'a demonstration of the person doing {}.',
        'a photo of a person during {}.',
        'a video of a person during {}.',
        'a example of a person during {}.',
        'a demonstration of a person during {}.',
        'a photo of the person during {}.',
        'a video of the person during {}.',
        'a example of the person during {}.',
        'a demonstration of the person during {}.',
        'a photo of a person performing {}.',
        'a video of a person performing {}.',
        'a example of a person performing {}.',
        'a demonstration of a person performing {}.',
        'a photo of the person performing {}.',
        'a video of the person performing {}.',
        'a example of the person performing {}.',
        'a demonstration of the person performing {}.',
        'a photo of a person practicing {}.',
        'a video of a person practicing {}.',
        'a example of a person practicing {}.',
        'a demonstration of a person practicing {}.',
        'a photo of the person practicing {}.',
        'a video of the person practicing {}.',
        'a example of the person practicing {}.',
        'a demonstration of the person practicing {}.',
    ]
}
