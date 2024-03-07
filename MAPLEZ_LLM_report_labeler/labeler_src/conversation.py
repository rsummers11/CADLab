# NIH Clinical Center
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
# Ricardo Bigolin Lanfredi - 2023-14-12
# Description:
# Auxiliary file used to create prompts for the LLM
# modified from https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py
# the main modification was to make the conversation header be specific to the task of labeling cxr reports

import dataclasses
from enum import auto, Enum
from typing import List, Tuple, Any

label_set_mimic =  ['atelectasis','cardiomegaly','consolidation', 'edema', 'enlarged cardiomediastinum','fracture','lung lesion','lung opacity', 'pleural effusion', 'pleural thickening', 'fibrosis', 'pneumonia', 'pneumothorax', 'medical equipment']
label_set_nih = ['atelectasis', 'cardiomegaly', 'consolidation', 'pneumothorax', 'pneumonia', 'pleural thickening', 'nodule', 'mass', 'infiltration', 'hernia', 'fibrosis', 'emphysema', 'effusion', 'edema']

class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None

    skip_next: bool = False
    conv_id: Any = None

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            conv_id=self.conv_id)

    def dict(self):
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
            "conv_id": self.conv_id,
        }

# answer_template = '[Absent, Present, Doubtly Present, or Unspecified"]'
# answer_template = '[Present/Absent], [Mentioned/Not Mentioned]'
# answer_template = '[Present/Possibly Present/Absent]'
# answer_template = """[\n
# 1: for when the report alone is sufficient to indicate that the radiologist observed evidence of a finding;\n
# 0: for when the report alone is sufficient to to rule out that the radiologist observed evidence of a finding;\n
# -1: for when the report is certainly directly related to a finding;\n
# n: for when the reported is completely unrelated to a finding;\n
# ]
# """
answer_template = '[Yes/No]'

# You are a helpful chatbot that can accuratly classify radiology reports for their \
#     sufficient information about presence and/or absence of the following findings:\n
#     * cardiac congestion,\n
#     * lung opacities (that includes pneumonia, atelectasis, dystelectasis and other airway processes),\n 
#     * pleural abnormalities,\n
#     * pneumothorax,\n
#     * presence of thoracic drains, venous catheters, gastric tubes, tracheal tubes, misplacement of any devices. 
#     Be mindful of medical synonyms and definitions of abnormalities.
#     Structure your answer like the following template:

# conv_v1_2 = Conversation(
#     system=f"""
#         You are a helpful chatbot, that can accuratly classify radiology reports for the presence or absence of fingins. 
#     Each report, you will classify for the presence or absence of the following findings: 
#     Cardiac congestion, lung opacities (that includes pneumonia, atelectasis, dystelectasis and other airway processes), 
#     pleural effusion (this does NOT inlcude pericardial effusion), pneumothorax, presence of thoracic drains, 
#     presence of venous catheters, presence of gastric tubes, presence of tracheal tubes, misplacement of any devices. 
#     structure your answer like the template I provide you and return this template
#     {template}
#     """,
#     roles=("Human", "Assistant"),
#     messages=(),
#     offset=0,
#     sep_style=SeparatorStyle.SINGLE,
#     sep="###",
# )


# conv_v1_2 = Conversation(
#     system="A chat between a radiologist and an artificial intelligence assistant trained to understand radiology reports and any synonyms and word equivalency of findings and medical terms that may appear in the report. "
#            f"The assistant gives helpful structured answers to the radiologist."
#            ,
#     roles=("Human", "Assistant"),
#     messages=(),
#     offset=2,
#     sep_style=SeparatorStyle.SINGLE,
#     sep="###",
# )

conv_v1_2 = Conversation(
    system="### System: A chat between a radiologist and an artificial intelligence assistant trained to understand radiology reports and any synonyms and word equivalency of findings and medical terms that may appear in the report. "
           f"The assistant gives helpful structured answers to the radiologist."
           ,
    roles=("User", "Assistant"),
    messages=(),
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_bair_v1 = Conversation(
    system="BEGINNING OF CONVERSATION:",
    roles=("USER", "GPT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)


default_conversation = conv_v1_2
conv_templates = {
    "v1": conv_v1_2,
    "bair_v1": conv_bair_v1,
}


if __name__ == "__main__":
    print(default_conversation.get_prompt())
