from typing import Dict, Optional, Union

import numpy as np

from .generation import codec_decode, generate_coarse, generate_fine, generate_text_semantic

# Import untuk dukungan bahasa Indonesia
try:
    from transformers import AutoProcessor, AutoModel
    import librosa
    import torch
    
    # Variabel global untuk model Indonesia
    _indonesian_processor = None
    _indonesian_model = None
    
    def _load_indonesian_model():
        """Load model MMS-TTS Indonesia"""
        global _indonesian_processor, _indonesian_model
        if _indonesian_processor is None or _indonesian_model is None:
            print("Loading Indonesian MMS-TTS model...")
            _indonesian_processor = AutoProcessor.from_pretrained("facebook/mms-tts-ind")
            _indonesian_model = AutoModel.from_pretrained("facebook/mms-tts-ind")
        return _indonesian_processor, _indonesian_model
    
    def _is_indonesian_text(text):
        """Deteksi apakah teks mengandung bahasa Indonesia"""
        indonesian_keywords = ['yang', 'dan', 'di', 'untuk', 'dengan', 'ini', 'itu', 'ada', 'adalah']
        words = text.lower().split()
        return any(keyword in words for keyword in indonesian_keywords)
    
    def _generate_indonesian_audio(text):
        """Generate audio untuk teks bahasa Indonesia"""
        processor, model = _load_indonesian_model()
        
        inputs = processor(text=text, return_tensors="pt")
        
        with torch.no_grad():
            output = model(**inputs).waveform
        
        # Konversi ke format Bark (24kHz)
        audio_array = output.squeeze().numpy()
        audio_array = librosa.resample(audio_array, orig_sr=16000, target_sr=24000)
        return audio_array
        
except ImportError:
    # Fallback jika dependencies tidak terinstall
    print("Warning: Indonesian language support dependencies not installed.")
    _is_indonesian_text = lambda text: False
    _generate_indonesian_audio = lambda text: None


def text_to_semantic(
    text: str,
    history_prompt: Optional[Union[Dict, str]] = None,
    temp: float = 0.7,
    silent: bool = False,
):
    """Generate semantic array from text.

    Args:
        text: text to be turned into audio
        history_prompt: history choice for audio cloning
        temp: generation temperature (1.0 more diverse, 0.0 more conservative)
        silent: disable progress bar

    Returns:
        numpy semantic array to be fed into `semantic_to_waveform`
    """
    # Deteksi bahasa Indonesia dan gunakan model Indonesia jika sesuai
    if _is_indonesian_text(text) and (history_prompt is None or "id_speaker" in str(history_prompt)):
        # Untuk bahasa Indonesia, kita tidak bisa menghasilkan semantic tokens dengan cara biasa
        # Jadi kita akan mengembalikan array kosong dan menangani di generate_audio
        return np.array([])
    
    x_semantic = generate_text_semantic(
        text,
        history_prompt=history_prompt,
        temp=temp,
        silent=silent,
        use_kv_caching=True
    )
    return x_semantic


def semantic_to_waveform(
    semantic_tokens: np.ndarray,
    history_prompt: Optional[Union[Dict, str]] = None,
    temp: float = 0.7,
    silent: bool = False,
    output_full: bool = False,
):
    """Generate audio array from semantic input.

    Args:
        semantic_tokens: semantic token output from `text_to_semantic`
        history_prompt: history choice for audio cloning
        temp: generation temperature (1.0 more diverse, 0.0 more conservative)
        silent: disable progress bar
        output_full: return full generation to be used as a history prompt

    Returns:
        numpy audio array at sample frequency 24khz
    """
    # Jika semantic_tokens kosong (bahasa Indonesia), langsung generate audio
    if len(semantic_tokens) == 0:
        # Untuk bahasa Indonesia, kita tidak bisa menghasilkan full generation
        if output_full:
            return {}, _generate_indonesian_audio("")  # Teks tidak tersedia di sini
        return _generate_indonesian_audio("")  # Teks tidak tersedia di sini
    
    coarse_tokens = generate_coarse(
        semantic_tokens,
        history_prompt=history_prompt,
        temp=temp,
        silent=silent,
        use_kv_caching=True
    )
    fine_tokens = generate_fine(
        coarse_tokens,
        history_prompt=history_prompt,
        temp=0.5,
    )
    audio_arr = codec_decode(fine_tokens)
    if output_full:
        full_generation = {
            "semantic_prompt": semantic_tokens,
            "coarse_prompt": coarse_tokens,
            "fine_prompt": fine_tokens,
        }
        return full_generation, audio_arr
    return audio_arr


def save_as_prompt(filepath, full_generation):
    assert(filepath.endswith(".npz"))
    assert(isinstance(full_generation, dict))
    assert("semantic_prompt" in full_generation)
    assert("coarse_prompt" in full_generation)
    assert("fine_prompt" in full_generation)
    np.savez(filepath, **full_generation)


def generate_audio(
    text: str,
    history_prompt: Optional[Union[Dict, str]] = None,
    text_temp: float = 0.7,
    waveform_temp: float = 0.7,
    silent: bool = False,
    output_full: bool = False,
):
    """Generate audio array from input text.

    Args:
        text: text to be turned into audio
        history_prompt: history choice for audio cloning
        text_temp: generation temperature (1.0 more diverse, 0.0 more conservative)
        waveform_temp: generation temperature (1.0 more diverse, 0.0 more conservative)
        silent: disable progress bar
        output_full: return full generation to be used as a history prompt

    Returns:
        numpy audio array at sample frequency 24khz
    """
    # Deteksi bahasa Indonesia
    is_indonesian = _is_indonesian_text(text)
    is_indonesian_speaker = history_prompt and "id_speaker" in str(history_prompt)
    
    # Gunakan model Indonesia jika sesuai
    if is_indonesian or is_indonesian_speaker:
        if not silent:
            print("Using Indonesian MMS-TTS model")
        
        audio_arr = _generate_indonesian_audio(text)
        
        if output_full:
            # Untuk model Indonesia, kita tidak memiliki full generation data
            full_generation = {
                "semantic_prompt": np.array([]),
                "coarse_prompt": np.array([]),
                "fine_prompt": np.array([]),
            }
            return full_generation, audio_arr
        return audio_arr
    
    # Gunakan model Bark asli untuk bahasa lain
    semantic_tokens = text_to_semantic(
        text,
        history_prompt=history_prompt,
        temp=text_temp,
        silent=silent,
    )
    out = semantic_to_waveform(
        semantic_tokens,
        history_prompt=history_prompt,
        temp=waveform_temp,
        silent=silent,
        output_full=output_full,
    )
    if output_full:
        full_generation, audio_arr = out
        return full_generation, audio_arr
    else:
        audio_arr = out
    return audio_arr
