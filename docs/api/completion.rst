ecgtizer.completion
===================

.. automodule:: ecgtizer.completion
   :members:
   :undoc-members:
   :show-inheritance:

Functional Examples
-------------------

Complete partial leads to 10 seconds
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The main function :func:`completion_` extends partial ECG recordings
(2.5 s or 5 s) to the full 10-second duration using a pre-trained
convolutional autoencoder:

.. code-block:: python

   import numpy as np
   from ecgtizer.completion import completion_

   # Simulate a 3x4 ECG with 2.5-second leads (1250 samples at 500 Hz)
   rng = np.random.RandomState(42)
   t = np.linspace(0, 2.5, 1250)
   ecg = {}
   for name in ["I", "II", "III", "AVR", "AVL", "AVF",
                 "V1", "V2", "V3", "V4", "V5", "V6"]:
       ecg[name] = (300 * np.sin(2 * np.pi * 1.2 * t) + rng.randn(1250) * 10)

   # Run completion
   completed = completion_(ecg, "model/Model_Completion.pth", device="cpu")

   # Each lead is now 5000 samples (10 seconds at 500 Hz)
   for name, signal in completed.items():
       print(f"{name}: {len(signal)} samples")

Complete from a real extraction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from ecgtizer import ECGtizer
   from ecgtizer.completion import completion_

   ecg = ECGtizer("data/PTB-XL/PDF/00075_hr.pdf", dpi=500)

   completed = completion_(
       ecg.extracted_lead,
       "model/Model_Completion.pth",
       device="cpu",
   )

   # Compare original vs completed lengths
   for name in ["I", "II", "V1"]:
       orig_len = len(ecg.extracted_lead[name])
       comp_len = len(completed[name])
       print(f"{name}: {orig_len} -> {comp_len} samples")

Load the model separately
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from ecgtizer.completion import load_model

   model = load_model("model/Model_Completion.pth", device="cpu")
   print(model)
   # Autoencoder_net(
   #   (conv1d_1): Convolution1D_layer(...)
   #   (conv2d_1): Convolution2D_layer(...)
   #   ...
   # )

   # Count parameters
   n_params = sum(p.numel() for p in model.parameters())
   print(f"Model has {n_params:,} parameters")

Normalize and denormalize signals
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import numpy as np
   from ecgtizer.completion import normalization2, denormalization

   signal = np.array([100, 200, 300, 400, 500], dtype=float)

   # Normalize to [-1, 1]
   normalized, orig_min, orig_max = normalization2(signal)
   print(normalized)   # [-1.0, -0.5, 0.0, 0.5, 1.0]

   # Denormalize back
   restored = denormalization(normalized, orig_min, orig_max)
   print(restored)     # [100. 200. 300. 400. 500.]

Resample a signal to 5000 points
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import numpy as np
   from ecgtizer.completion import linear_interpolation

   # Short signal (1250 points = 2.5 seconds at 500 Hz)
   short_signal = np.sin(np.linspace(0, 4 * np.pi, 1250))

   # Resample to 5000 points (10 seconds)
   long_signal = linear_interpolation(short_signal)
   print(long_signal.shape)   # (5000,)

Prepare input matrix for the model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import numpy as np
   from ecgtizer.completion import replace_random

   # 12 leads, each 512 samples (model input size)
   data = np.random.randn(12, 512).astype(np.float32)

   # Mark leads 6-11 as missing (typical 3x4 with 2.5 s per lead)
   data_prepared, scale = replace_random(data, True_data=True)
   print(data_prepared.shape)   # (12, 512)
   print(scale.shape)           # (12, 512) — binary mask of valid data
