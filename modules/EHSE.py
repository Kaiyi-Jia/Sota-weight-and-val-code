import torch
import torch.nn as nn
import clip
from transformers import CLIPProcessor, CLIPModel


class EHSE(nn.Module):
    """
    Expression High-level Semantic Encoding (EHSE) Module
    """

    def __init__(self, num_classes=8, class_names=None):
        super().__init__()
        self.num_classes = num_classes
        self.class_names = class_names if class_names else ['Neutral', 'Happy', 'Sad', 'Surprised', 'Scared', 'Disgusted',
                                                            'Angry', 'Contemptuous']

        # 5 Regions of Interest (ROIs)
        self.rois = ['eyebrows', 'eyes', 'mouth', 'nose', 'cheeks']

        # FACS physiological feature description library
        self.roi_phrases = {
            'Neutral': {
                'eyebrows': "straight and relaxed eyebrows without any arching or furrowing",
                'eyes': "naturally open eyes with a neutral gaze that are neither narrowed nor widened",
                'mouth': "a naturally closed mouth with relaxed lips that avoids smiling or frowning",
                'nose': "a relaxed nose without any wrinkling or flaring",
                'cheeks': "calm cheeks that are not raised or tensed"
            },
            'Happy': {
                'eyebrows': "slightly raised eyebrows without any angry furrowing",
                'eyes': "slightly narrowed eyes with distinctive crow's feet wrinkles at the corners and raised lower eyelids",
                'mouth': "upward and backward pulled mouth corners forming a smile that may part the lips to reveal teeth",
                'nose': "a relaxed and smooth nose structure",
                'cheeks': "prominently raised cheeks forming pronounced nasolabial folds"
            },
            'Sad': {
                'eyebrows': "inner corners of the eyebrows raised and drawn together tightly to form an inverted V shape",
                'eyes': "droopy upper eyelids and slightly raised lower eyelids with a downward gaze",
                'mouth': "downward pulled mouth corners with the lower lip pushed slightly upward",
                'nose': "a nose area showing slight expressions of wrinkling",
                'cheeks': "relaxed and slightly pulled down cheeks feeling heavy"
            },
            'Surprised': {
                'eyebrows': "highly raised and curved eyebrows that form a high distinct arch",
                'eyes': "widened eyes with highly raised upper eyelids and tensed lower eyelids exposing the sclera",
                'mouth': "an open mouth with parted lips and a dropped jaw forming a clear oval shape",
                'nose': "a relaxed nose without any micro-wrinkles",
                'cheeks': "relaxed and vertically stretched cheek muscles"
            },
            'Scared': {
                'eyebrows': "raised and tightly pulled together eyebrows forming a flattened arch",
                'eyes': "highly raised upper eyelids and tensed, elevated lower eyelids making the eyes appear wide and staring",
                'mouth': "mouth corners pulled straight back towards the ears with horizontally stretched lips",
                'nose': "a nose showing slight tension and subtle wrinkling",
                'cheeks': "highly raised and tensed cheek muscles reacting to terror"
            },
            'Disgusted': {
                'eyebrows': "lowered eyebrows with inner corners drawn downwards strongly",
                'eyes': "partially closed upper eyelids with raised lower eyelids creating a squint",
                'mouth': "mouth corners pulled down and back with an elevated upper lip",
                'nose': "a heavily wrinkled nose bridge due to strong levator labii superioris action",
                'cheeks': "raised cheeks forming prominent folds and bags under the eyes"
            },
            'Angry': {
                'eyebrows': "lowered and drawn together eyebrows forming a sharp vertical furrow between them",
                'eyes': "tense upper eyelids and raised lower eyelids creating a fierce glaring look",
                'mouth': "lips pressed firmly and tightly together or parted with tightly clenched teeth in a squared shape",
                'nose': "a nose displaying clear flaring of the nostrils",
                'cheeks': "intensely tensed and rigid cheek muscles"
            },
            'Contemptuous': {
                'eyebrows': "neutral or slightly raised asymmetrical eyebrows",
                'eyes': "neutral eyes that may hint at a subtle side-roll",
                'mouth': "a single mouth corner pulled uniquely up and back forming an asymmetrical sneer",
                'nose': "a completely relaxed nose structure",
                'cheeks': "neutral and relaxed cheeks on one side while slightly tensed on the active side"
            }
        }

    def _generate_paragraph_texts(self):
        """
        1. Global Text: Coherent, fluent paragraph-level description prototypes (t_global)
        2. Local Texts: Fine-grained description lists for each individual ROI (t_local)
        """
        global_prompts = []
        local_prompts_per_class = []  # Each class contains 5 regional descriptions

        for c in range(self.num_classes):
            emotion = self.class_names[c]
            p = self.roi_phrases.get(emotion, self.roi_phrases['Neutral'])

            # 1. Paragraph-style natural language descriptions
            paragraph = f"A {emotion.lower()} expression features {p['mouth']}, {p['cheeks']}, {p['eyes']}, {p['nose']}, and {p['eyebrows']}."
            global_prompts.append(paragraph)

            # 2. Local region descriptions
            roi_sentences = [f"The {roi} in a {emotion.lower()} expression shows {p[roi]}." for roi in self.rois]
            local_prompts_per_class.append(roi_sentences)

        return global_prompts, local_prompts_per_class

    def get_semantic_embeddings(self, clip_model, device):
        global_texts, local_texts_batch = self._generate_paragraph_texts()

        global_inputs = clip.tokenize(global_texts, truncate=True).to(device)
        with torch.no_grad():
            t_global = clip_model.encode_text(global_inputs)
            t_global = t_global / t_global.norm(dim=-1, keepdim=True)

        t_local_list = []
        for class_roi_texts in local_texts_batch:
            local_inputs = clip.tokenize(class_roi_texts, truncate=True).to(device)
            with torch.no_grad():
                roi_embeds = clip_model.encode_text(local_inputs)
                mean_roi_embed = roi_embeds.mean(dim=0)
                mean_roi_embed = mean_roi_embed / mean_roi_embed.norm(dim=-1, keepdim=True)
                t_local_list.append(mean_roi_embed)

        t_local = torch.stack(t_local_list, dim=0)

        return t_global.float(), t_local.float()
