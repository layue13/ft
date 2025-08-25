
import sys
import os
sys.path.append('llama.cpp')

from convert import convert_hf_to_gguf

# 转换模型
convert_hf_to_gguf(
    model_path='./gemma3-1b-tool-use-merged',
    output_path='./gemma3-1b-tool-use.gguf',
    model_type='llama'
)
