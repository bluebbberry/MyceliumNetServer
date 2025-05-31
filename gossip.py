import time
import threading
import logging
from network import PeerNetwork
from train import MusicRecommendationModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GossipLearning:
    def __init__(self, host='localhost', port=8000, peers=None):
        self.network = PeerNetwork(host, port, peers)
        self.model = MusicRecommendationModel()
        self.is_training = False
        self.gossip_rounds = 0
        self.max_rounds = 50
        self.gossip_interval = 10  # seconds

        # Set message handler for network
        self.network.set_message_handler(self._handle_message)

    def _handle_message(self, message, addr):
        """Handle incoming GOSSIP messages"""
        try:
            if message['type'] == 'parameter_request':
                # Send our current parameters
                params = self.model.get_parameters()
                if params:
                    return {
                        'type': 'parameter_response',
                        'parameters': params,
                        'round': self.gossip_rounds
                    }
                else:
                    return {
                        'type': 'parameter_response',
                        'parameters': None,
                        'round': self.gossip_rounds
                    }

            elif message['type'] == 'parameter_exchange':
                # Received parameters from another peer
                peer_params = message.get('parameters')
                if peer_params:
                    success = self.model.update_parameters(peer_params)
                    logger.info(f"Parameter exchange from {addr}: {'success' if success else 'failed'}")

                # Send our parameters back
                our_params = self.model.get_parameters()
                return {
                    'type': 'parameter_response',
                    'parameters': our_params,
                    'round': self.gossip_rounds
                }

        except Exception as e:
            logger.error(f"Error handling message: {e}")

        return None

    def start_network(self):
        """Start the network server"""
        self.network.start_server_thread()
        logger.info(f"GOSSIP node started on {self.network.host}:{self.network.port}")

    def load_data(self, csv_file):
        """Load training data"""
        if self.model.load_data(csv_file):
            if self.model.prepare_data():
                return self.model.train_local()
        return False

    def start_gossip_training(self, max_rounds=50):
        """Start GOSSIP training process"""
        if not self.model.is_fitted:
            logger.error("Model not trained on local data. Load data first.")
            return False

        self.max_rounds = max_rounds
        self.is_training = True
        self.gossip_rounds = 0

        logger.info(f"Starting GOSSIP training for {max_rounds} rounds")

        # Start GOSSIP in separate thread
        gossip_thread = threading.Thread(target=self._gossip_loop)
        gossip_thread.daemon = True
        gossip_thread.start()

        return True

    def _gossip_loop(self):
        """Main GOSSIP training loop"""
        while self.is_training and self.gossip_rounds < self.max_rounds:
            try:
                # Get random peer
                peer = self.network.get_random_peer()
                if not peer:
                    logger.warning("No peers available for GOSSIP")
                    time.sleep(self.gossip_interval)
                    continue

                peer_host, peer_port = peer
                logger.info(f"GOSSIP round {self.gossip_rounds + 1}: contacting {peer_host}:{peer_port}")

                # Send parameter exchange request
                our_params = self.model.get_parameters()
                if our_params:
                    message = {
                        'type': 'parameter_exchange',
                        'parameters': our_params,
                        'round': self.gossip_rounds
                    }

                    response = self.network.send_message(peer_host, peer_port, message)

                    if response and response.get('parameters'):
                        # Update our parameters with received ones
                        self.model.update_parameters(response['parameters'])
                        logger.info(f"GOSSIP round {self.gossip_rounds + 1} completed successfully")
                    else:
                        logger.warning(f"No response from {peer_host}:{peer_port}")

                self.gossip_rounds += 1

                # Wait before next round
                if self.gossip_rounds < self.max_rounds:
                    time.sleep(self.gossip_interval)

            except Exception as e:
                logger.error(f"Error in GOSSIP round {self.gossip_rounds + 1}: {e}")
                time.sleep(self.gossip_interval)

        self.is_training = False
        logger.info(f"GOSSIP training completed after {self.gossip_rounds} rounds")

    def stop_training(self):
        """Stop GOSSIP training"""
        self.is_training = False
        logger.info("GOSSIP training stopped")

    def get_training_status(self):
        """Get current training status"""
        return {
            'is_training': self.is_training,
            'current_round': self.gossip_rounds,
            'max_rounds': self.max_rounds,
            'model_fitted': self.model.is_fitted
        }

    def save_model(self, filename):
        """Save the trained model"""
        return self.model.save_model(filename)

    def load_model(self, filename):
        """Load a saved model"""
        return self.model.load_model(filename)

    def predict_rating(self, user_id, song_id):
        """Predict rating for user-song pair"""
        return self.model.predict_rating(user_id, song_id)

    def recommend_songs(self, user_id, n_recommendations=5):
        """Get song recommendations for user"""
        return self.model.recommend_songs(user_id, n_recommendations)

    def stop_network(self):
        """Stop the network server"""
        self.network.stop_server()

    def wait_for_training_completion(self):
        """Wait for GOSSIP training to complete"""
        while self.is_training:
            time.sleep(1)

    def manual_gossip_round(self):
        """Perform a single GOSSIP round manually"""
        if not self.model.is_fitted:
            logger.error("Model not fitted")
            return False

        peer = self.network.get_random_peer()
        if not peer:
            logger.warning("No peers available")
            return False

        peer_host, peer_port = peer
        our_params = self.model.get_parameters()

        if our_params:
            message = {
                'type': 'parameter_exchange',
                'parameters': our_params,
                'round': self.gossip_rounds
            }

            response = self.network.send_message(peer_host, peer_port, message)

            if response and response.get('parameters'):
                self.model.update_parameters(response['parameters'])
                self.gossip_rounds += 1
                logger.info(f"Manual GOSSIP round completed (round {self.gossip_rounds})")
                return True

        return False