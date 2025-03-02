import tenseal as ts
import pickle
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEncryption:
    def __init__(self):
        self.context = self._generate_context()

    def _generate_context(self):
        try:
            context = ts.Context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=8192,
                coeff_mod_bit_sizes=[60, 40, 40, 40]
            )
            context.global_scale = 2**40
            context.generate_galois_keys()
            return context
        except Exception as e:
            logger.error(f"Error generating encryption context: {e}")
            raise

    def save_context(self, context_path):
        try:
            with open(context_path, 'wb') as f:
                f.write(self.context.serialize(save_secret_key=True))
            logger.info("Encryption context saved successfully")
        except Exception as e:
            logger.error(f"Error saving encryption context: {e}")
            raise

    def load_context(self, context_path):
        try:
            with open(context_path, 'rb') as f:
                context_bytes = f.read()
            self.context = ts.Context.load(context_bytes)
            logger.info("Encryption context loaded successfully")
        except Exception as e:
            logger.error(f"Error loading encryption context: {e}")
            raise
