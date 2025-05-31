import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MusicRecommendationModel:
    def __init__(self, n_components=10, random_state=42):
        self.n_components = n_components
        self.model = NMF(n_components=n_components, random_state=random_state, max_iter=100)
        self.user_mapping = {}
        self.song_mapping = {}
        self.user_factors = None
        self.song_factors = None
        self.is_fitted = False

    def load_data(self, csv_file):
        """Load rating data from CSV file"""
        try:
            self.data = pd.read_csv(csv_file)
            logger.info(f"Loaded {len(self.data)} ratings from {csv_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return False

    def prepare_data(self):
        """Convert rating data to user-item matrix"""
        try:
            # Create mappings for users and songs
            unique_users = sorted(self.data['user_id'].unique())
            unique_songs = sorted(self.data['song_id'].unique())

            self.user_mapping = {user: idx for idx, user in enumerate(unique_users)}
            self.song_mapping = {song: idx for idx, song in enumerate(unique_songs)}

            # Create user-item matrix
            n_users = len(unique_users)
            n_songs = len(unique_songs)

            self.user_item_matrix = np.zeros((n_users, n_songs))

            for _, row in self.data.iterrows():
                user_idx = self.user_mapping[row['user_id']]
                song_idx = self.song_mapping[row['song_id']]
                self.user_item_matrix[user_idx, song_idx] = row['rating']

            logger.info(f"Created {n_users}x{n_songs} user-item matrix")
            return True

        except Exception as e:
            logger.error(f"Failed to prepare data: {e}")
            return False

    def train_local(self):
        """Train the model on local data"""
        try:
            if not hasattr(self, 'user_item_matrix'):
                logger.error("Data not prepared. Call prepare_data() first.")
                return False

            # Fit NMF model
            self.user_factors = self.model.fit_transform(self.user_item_matrix)
            self.song_factors = self.model.components_
            self.is_fitted = True

            logger.info(f"Model trained with {self.n_components} components")
            return True

        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False

    def get_parameters(self):
        """Get model parameters for GOSSIP exchange"""
        if not self.is_fitted:
            return None

        return {
            'user_factors': self.user_factors.tolist(),
            'song_factors': self.song_factors.tolist(),
            'user_mapping': self.user_mapping,
            'song_mapping': self.song_mapping,
            'n_components': self.n_components
        }

    def update_parameters(self, new_params):
        """Update model parameters from GOSSIP exchange"""
        try:
            if not new_params:
                return False

            # Convert lists back to numpy arrays
            new_user_factors = np.array(new_params['user_factors'])
            new_song_factors = np.array(new_params['song_factors'])

            if self.is_fitted:
                # Average with current parameters
                self.user_factors = (self.user_factors + new_user_factors) / 2.0
                self.song_factors = (self.song_factors + new_song_factors) / 2.0
            else:
                # First time receiving parameters
                self.user_factors = new_user_factors
                self.song_factors = new_song_factors
                self.user_mapping = new_params['user_mapping']
                self.song_mapping = new_params['song_mapping']
                self.n_components = new_params['n_components']
                self.is_fitted = True

            logger.info("Parameters updated via GOSSIP")
            return True

        except Exception as e:
            logger.error(f"Failed to update parameters: {e}")
            return False

    def predict_rating(self, user_id, song_id):
        """Predict rating for a user-song pair"""
        if not self.is_fitted:
            logger.error("Model not trained")
            return None

        try:
            if user_id not in self.user_mapping or song_id not in self.song_mapping:
                logger.warning(f"User {user_id} or song {song_id} not in training data")
                return None

            user_idx = self.user_mapping[user_id]
            song_idx = self.song_mapping[song_id]

            predicted_rating = np.dot(self.user_factors[user_idx], self.song_factors[:, song_idx])
            return float(predicted_rating)

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return None

    def recommend_songs(self, user_id, n_recommendations=5):
        """Recommend top N songs for a user"""
        if not self.is_fitted:
            logger.error("Model not trained")
            return []

        try:
            if user_id not in self.user_mapping:
                logger.warning(f"User {user_id} not in training data")
                return []

            user_idx = self.user_mapping[user_id]

            # Calculate predicted ratings for all songs
            user_vector = self.user_factors[user_idx]
            predicted_ratings = np.dot(user_vector, self.song_factors)

            # Get top N songs
            song_indices = np.argsort(predicted_ratings)[::-1][:n_recommendations]

            # Convert back to song IDs
            reverse_song_mapping = {idx: song for song, idx in self.song_mapping.items()}
            recommendations = [
                {
                    'song_id': reverse_song_mapping[idx],
                    'predicted_rating': float(predicted_ratings[idx])
                }
                for idx in song_indices
            ]

            return recommendations

        except Exception as e:
            logger.error(f"Recommendation failed: {e}")
            return []

    def save_model(self, filename):
        """Save model to pickle file"""
        try:
            model_data = {
                'user_factors': self.user_factors,
                'song_factors': self.song_factors,
                'user_mapping': self.user_mapping,
                'song_mapping': self.song_mapping,
                'n_components': self.n_components,
                'is_fitted': self.is_fitted
            }

            with open(filename, 'wb') as f:
                pickle.dump(model_data, f)

            logger.info(f"Model saved to {filename}")
            return True

        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False

    def load_model(self, filename):
        """Load model from pickle file"""
        try:
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)

            self.user_factors = model_data['user_factors']
            self.song_factors = model_data['song_factors']
            self.user_mapping = model_data['user_mapping']
            self.song_mapping = model_data['song_mapping']
            self.n_components = model_data['n_components']
            self.is_fitted = model_data['is_fitted']

            logger.info(f"Model loaded from {filename}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False