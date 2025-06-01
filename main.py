#!/usr/bin/env python3
"""
Enhanced Synchronized Gossip Learning for Music Recommendations
Now includes song attributes and content-based filtering
"""

import json
import sys
import time
import threading
import socket
import random
import pickle
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


@dataclass
class Config:
    """Simple configuration class"""
    host: str = "localhost"
    port: int = 8000
    peers: List[Tuple[str, int]] = None
    ratings_file: str = "ratings.csv"
    songs_file: str = "songs.csv"
    gossip_rounds: int = 10
    gossip_interval: float = 2.0
    peer_check_interval: float = 1.0
    max_wait_time: float = 60.0

    @classmethod
    def load(cls, filename="config.json"):
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            return cls(
                host=data.get("host", "localhost"),
                port=data.get("port", 8000),
                peers=[tuple(p) for p in data.get("peers", [])],
                ratings_file=data.get("ratings_file", "ratings.csv"),
                songs_file=data.get("songs_file", "songs.csv"),
                gossip_rounds=data.get("gossip_rounds", 10),
                gossip_interval=data.get("gossip_interval", 2.0),
                peer_check_interval=data.get("peer_check_interval", 1.0),
                max_wait_time=data.get("max_wait_time", 60.0)
            )
        except Exception as e:
            print(f"Error loading config: {e}")
            return cls()


class EnhancedMusicModel:
    """Enhanced music recommendation model using collaborative + content-based filtering"""

    def __init__(self):
        # User-based data
        self.user_ratings = {}  # user_id -> {song_id: rating}
        self.user_averages = {}
        self.user_preferences = {}  # user_id -> preference vector based on song attributes

        # Song-based data
        self.song_attributes = {}  # song_id -> attribute vector
        self.song_averages = {}

        # Model parameters
        self.global_average = 0.0
        self.attribute_weights = {}  # learned weights for different attributes
        self.is_trained = False

        # Attribute names for normalization
        self.audio_features = ['acousticness', 'danceability', 'energy', 'instrumentalness',
                               'liveness', 'loudness', 'speechiness', 'tempo']
        self.categorical_features = ['genre', 'key', 'mode', 'time_signature']

    def load_data(self, ratings_file, songs_file):
        """Load ratings and song attributes data"""
        try:
            # Load ratings
            ratings_df = pd.read_csv(ratings_file)
            for _, row in ratings_df.iterrows():
                user_id = int(row['user_id'])
                song_id = int(row['song_id'])
                rating = float(row['rating'])

                if user_id not in self.user_ratings:
                    self.user_ratings[user_id] = {}
                self.user_ratings[user_id][song_id] = rating

            # Load song attributes
            songs_df = pd.read_csv(songs_file)
            for _, row in songs_df.iterrows():
                song_id = int(row['song_id'])

                # Create attribute vector
                attributes = {}
                for feature in self.audio_features:
                    attributes[feature] = float(row[feature])

                # Handle categorical features
                attributes['genre'] = str(row['genre'])
                attributes['key'] = int(row['key'])
                attributes['mode'] = int(row['mode'])
                attributes['time_signature'] = int(row['time_signature'])
                attributes['duration_ms'] = int(row['duration_ms'])

                # Store metadata
                attributes['song_title'] = str(row['song_title'])
                attributes['artist'] = str(row['artist'])

                self.song_attributes[song_id] = attributes

            print(f"Loaded {len(self.user_ratings)} users and {len(self.song_attributes)} songs")
            self.train()
            return True

        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def _normalize_audio_features(self):
        """Normalize audio features to 0-1 range"""
        if not self.song_attributes:
            return

        # Get all audio feature values
        feature_values = {feature: [] for feature in self.audio_features}
        for song_attrs in self.song_attributes.values():
            for feature in self.audio_features:
                feature_values[feature].append(song_attrs[feature])

        # Calculate min/max for normalization
        feature_ranges = {}
        for feature, values in feature_values.items():
            feature_ranges[feature] = (min(values), max(values))

        # Normalize features
        for song_id, attrs in self.song_attributes.items():
            for feature in self.audio_features:
                min_val, max_val = feature_ranges[feature]
                if max_val > min_val:
                    normalized = (attrs[feature] - min_val) / (max_val - min_val)
                    self.song_attributes[song_id][feature + '_normalized'] = normalized
                else:
                    self.song_attributes[song_id][feature + '_normalized'] = 0.5

    def _compute_user_preferences(self):
        """Compute user preference vectors based on their rated songs"""
        self.user_preferences = {}

        for user_id, ratings in self.user_ratings.items():
            preferences = {}

            # Initialize preference sums
            for feature in self.audio_features:
                preferences[feature] = 0.0
            for feature in self.categorical_features:
                preferences[feature] = {}

            total_weight = 0.0

            # Weight by rating (higher rated songs contribute more to preferences)
            for song_id, rating in ratings.items():
                if song_id in self.song_attributes:
                    weight = (rating - 1) / 4.0  # Convert 1-5 rating to 0-1 weight
                    song_attrs = self.song_attributes[song_id]

                    # Audio features
                    for feature in self.audio_features:
                        normalized_feature = feature + '_normalized'
                        if normalized_feature in song_attrs:
                            preferences[feature] += song_attrs[normalized_feature] * weight

                    # Categorical features
                    for feature in self.categorical_features:
                        value = song_attrs[feature]
                        if value not in preferences[feature]:
                            preferences[feature][value] = 0.0
                        preferences[feature][value] += weight

                    total_weight += weight

            # Normalize preferences
            if total_weight > 0:
                for feature in self.audio_features:
                    preferences[feature] /= total_weight

                # For categorical features, convert to most preferred values
                for feature in self.categorical_features:
                    if preferences[feature]:
                        most_preferred = max(preferences[feature].items(), key=lambda x: x[1])
                        preferences[feature] = most_preferred[0]
                    else:
                        preferences[feature] = None

            self.user_preferences[user_id] = preferences

    def train(self):
        """Train the enhanced model"""
        if not self.user_ratings or not self.song_attributes:
            return False

        # Normalize audio features
        self._normalize_audio_features()

        # Compute basic averages (collaborative filtering part)
        for user_id, ratings in self.user_ratings.items():
            self.user_averages[user_id] = sum(ratings.values()) / len(ratings)

        # Compute song averages
        song_ratings = {}
        for user_id, ratings in self.user_ratings.items():
            for song_id, rating in ratings.items():
                if song_id not in song_ratings:
                    song_ratings[song_id] = []
                song_ratings[song_id].append(rating)

        for song_id, ratings in song_ratings.items():
            self.song_averages[song_id] = sum(ratings) / len(ratings)

        # Global average
        all_ratings = []
        for ratings in self.user_ratings.values():
            all_ratings.extend(ratings.values())
        self.global_average = sum(all_ratings) / len(all_ratings)

        # Compute user preferences (content-based part)
        self._compute_user_preferences()

        # Initialize attribute weights
        for feature in self.audio_features:
            self.attribute_weights[feature] = 1.0  # Equal weight initially
        for feature in self.categorical_features:
            self.attribute_weights[feature] = 1.0

        self.is_trained = True
        print("✅ Enhanced model training completed")
        return True

    def _content_similarity(self, user_id, song_id):
        """Calculate content-based similarity between user preferences and song"""
        if user_id not in self.user_preferences or song_id not in self.song_attributes:
            return 0.5  # Neutral similarity

        user_prefs = self.user_preferences[user_id]
        song_attrs = self.song_attributes[song_id]

        similarity = 0.0
        total_weight = 0.0

        # Audio feature similarity
        for feature in self.audio_features:
            if feature in user_prefs:
                normalized_feature = feature + '_normalized'
                if normalized_feature in song_attrs:
                    # Calculate similarity (1 - absolute difference)
                    diff = abs(user_prefs[feature] - song_attrs[normalized_feature])
                    feature_similarity = 1.0 - diff
                    weight = self.attribute_weights.get(feature, 1.0)
                    similarity += feature_similarity * weight
                    total_weight += weight

        # Categorical feature similarity
        for feature in self.categorical_features:
            if feature in user_prefs and user_prefs[feature] is not None:
                weight = self.attribute_weights.get(feature, 1.0)
                if song_attrs[feature] == user_prefs[feature]:
                    similarity += 1.0 * weight
                else:
                    similarity += 0.0 * weight
                total_weight += weight

        return similarity / total_weight if total_weight > 0 else 0.5

    def predict(self, user_id, song_id):
        """Enhanced prediction combining collaborative and content-based filtering"""
        if not self.is_trained:
            return None

        # Collaborative filtering component
        user_avg = self.user_averages.get(user_id, self.global_average)
        song_avg = self.song_averages.get(song_id, self.global_average)
        collaborative_pred = (user_avg + song_avg + self.global_average) / 3.0

        # Content-based filtering component
        content_similarity = self._content_similarity(user_id, song_id)
        content_pred = self.global_average + (content_similarity - 0.5) * 2.0  # Scale similarity to rating range

        # Combine predictions (weighted average)
        # Give more weight to collaborative if we have user history, otherwise rely more on content
        if user_id in self.user_ratings and len(self.user_ratings[user_id]) > 5:
            # User has enough history, trust collaborative filtering more
            prediction = 0.7 * collaborative_pred + 0.3 * content_pred
        else:
            # New user or limited history, rely more on content-based
            prediction = 0.3 * collaborative_pred + 0.7 * content_pred

        return max(1.0, min(5.0, prediction))  # Clamp to 1-5 range

    def get_song_info(self, song_id):
        """Get song information for display"""
        if song_id in self.song_attributes:
            attrs = self.song_attributes[song_id]
            return f"{attrs['song_title']} by {attrs['artist']} ({attrs['genre']})"
        return f"Song {song_id}"

    def get_parameters(self):
        """Get model parameters for gossip"""
        return {
            'user_averages': dict(self.user_averages),
            'song_averages': dict(self.song_averages),
            'global_average': float(self.global_average),
            'user_preferences': dict(self.user_preferences),
            'attribute_weights': dict(self.attribute_weights)
        }

    def update_parameters(self, other_params):
        """Average our parameters with another node's parameters"""
        if not other_params:
            return False

        try:
            # Average user averages
            for user_id, avg in other_params.get('user_averages', {}).items():
                user_id = int(user_id)
                if user_id in self.user_averages:
                    self.user_averages[user_id] = (self.user_averages[user_id] + avg) / 2.0
                else:
                    self.user_averages[user_id] = avg

            # Average song averages
            for song_id, avg in other_params.get('song_averages', {}).items():
                song_id = int(song_id)
                if song_id in self.song_averages:
                    self.song_averages[song_id] = (self.song_averages[song_id] + avg) / 2.0
                else:
                    self.song_averages[song_id] = avg

            # Average global average
            other_global = other_params.get('global_average', self.global_average)
            self.global_average = (self.global_average + other_global) / 2.0

            # Update user preferences (simple averaging for now)
            other_prefs = other_params.get('user_preferences', {})
            for user_id, prefs in other_prefs.items():
                user_id = int(user_id)
                if user_id in self.user_preferences:
                    # Average numerical preferences
                    for feature in self.audio_features:
                        if feature in prefs and feature in self.user_preferences[user_id]:
                            current = self.user_preferences[user_id][feature]
                            other = prefs[feature]
                            self.user_preferences[user_id][feature] = (current + other) / 2.0
                else:
                    self.user_preferences[user_id] = prefs

            # Average attribute weights
            other_weights = other_params.get('attribute_weights', {})
            for feature, weight in other_weights.items():
                if feature in self.attribute_weights:
                    self.attribute_weights[feature] = (self.attribute_weights[feature] + weight) / 2.0
                else:
                    self.attribute_weights[feature] = weight

            return True
        except Exception as e:
            print(f"Error updating parameters: {e}")
            return False


class SimpleNetwork:
    """Simple TCP networking without complex serialization"""

    def __init__(self, host, port, peers):
        self.host = host
        self.port = port
        self.peers = peers
        self.running = False
        self.server_socket = None
        self.message_handler = None

    def set_handler(self, handler):
        self.message_handler = handler

    def start_server(self):
        """Start server in background thread"""

        def server_loop():
            try:
                self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.server_socket.bind((self.host, self.port))
                self.server_socket.listen(5)
                self.running = True
                print(f"Server listening on {self.host}:{self.port}")

                while self.running:
                    try:
                        client, addr = self.server_socket.accept()
                        threading.Thread(target=self._handle_client, args=(client, addr), daemon=True).start()
                    except:
                        if self.running:
                            print("Server accept error")
                        break
            except Exception as e:
                print(f"Server error: {e}")

        threading.Thread(target=server_loop, daemon=True).start()
        time.sleep(0.5)  # Give server time to start

    def _handle_client(self, client, addr):
        """Handle incoming connection"""
        try:
            data = client.recv(8192)  # Increased buffer size for larger parameters
            if data:
                message = pickle.loads(data)
                if self.message_handler:
                    response = self.message_handler(message)
                    if response:
                        client.send(pickle.dumps(response))
        except Exception as e:
            print(f"Client handler error: {e}")
        finally:
            client.close()

    def send_message(self, peer_host, peer_port, message):
        """Send message to peer"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10)  # Increased timeout for larger data
            sock.connect((peer_host, peer_port))

            sock.send(pickle.dumps(message))
            response_data = sock.recv(8192)  # Increased buffer size
            if response_data:
                return pickle.loads(response_data)

        except Exception as e:
            print(f"Send error to {peer_host}:{peer_port} - {e}")
            return None
        finally:
            try:
                sock.close()
            except:
                pass
        return None

    def check_peer_availability(self, peer_host, peer_port):
        """Check if a peer is available and ready"""
        message = {'type': 'ping'}
        response = self.send_message(peer_host, peer_port, message)
        return response is not None and response.get('type') == 'pong'

    def get_random_peer(self):
        """Get random peer (not self)"""
        available = [(h, p) for h, p in self.peers if not (h == self.host and p == self.port)]
        return random.choice(available) if available else None

    def stop(self):
        """Stop server"""
        self.running = False
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass


class EnhancedGossipNode:
    """Enhanced gossip learning node with song attributes"""

    def __init__(self, config):
        self.config = config
        self.model = EnhancedMusicModel()
        self.network = SimpleNetwork(config.host, config.port, config.peers)
        self.network.set_handler(self._handle_message)
        self.training = False
        self.round_count = 0
        self.ready_for_training = False

    def _handle_message(self, message):
        """Handle incoming gossip message"""
        msg_type = message.get('type')

        if msg_type == 'ping':
            if self.ready_for_training:
                return {'type': 'pong', 'ready': True}
            else:
                return {'type': 'pong', 'ready': False}

        elif msg_type == 'gossip_request':
            return {
                'type': 'gossip_response',
                'parameters': self.model.get_parameters()
            }

        elif msg_type == 'predict_request':
            user_id = message.get('user_id')
            song_id = message.get('song_id')
            if user_id is not None and song_id is not None:
                prediction = self.model.predict(user_id, song_id)
                return {
                    'type': 'predict_response',
                    'prediction': prediction,
                    'song_info': self.model.get_song_info(song_id)
                }
        return None

    def wait_for_peers(self):
        """Wait until all peers are available and ready for training"""
        print("⏳ Waiting for all peers to be ready for training...")

        peer_list = [(h, p) for h, p in self.config.peers if not (h == self.config.host and p == self.config.port)]

        if not peer_list:
            print("⚠️ No peers configured, starting training immediately")
            return True

        start_time = time.time()

        while time.time() - start_time < self.config.max_wait_time:
            ready_peers = []
            unavailable_peers = []

            for peer_host, peer_port in peer_list:
                if self.network.check_peer_availability(peer_host, peer_port):
                    message = {'type': 'ping'}
                    response = self.network.send_message(peer_host, peer_port, message)
                    if response and response.get('ready', False):
                        ready_peers.append(f"{peer_host}:{peer_port}")
                    else:
                        unavailable_peers.append(f"{peer_host}:{peer_port} (not ready)")
                else:
                    unavailable_peers.append(f"{peer_host}:{peer_port} (offline)")

            if len(ready_peers) == len(peer_list):
                print(f"✅ All {len(peer_list)} peers are ready!")
                return True

            print(f"⏳ {len(ready_peers)}/{len(peer_list)} peers ready. Waiting for: {', '.join(unavailable_peers)}")
            time.sleep(self.config.peer_check_interval)

        print(f"⚠️ Timeout after {self.config.max_wait_time}s. Starting with available peers.")
        return False

    def start(self):
        """Start the node"""
        print(f"🚀 Starting enhanced node {self.config.host}:{self.config.port}")

        self.network.start_server()

        if not self.model.load_data(self.config.ratings_file, self.config.songs_file):
            print(f"❌ Failed to load data files")
            return False

        print(f"✅ Loaded data and trained enhanced model")
        self.ready_for_training = True
        print(f"🔵 Node is ready for training")

        return True

    def run_gossip(self):
        """Run gossip training after waiting for peers"""
        if not self.model.is_trained:
            print("❌ Model not trained")
            return

        self.wait_for_peers()

        print(f"🔄 Starting synchronized gossip for {self.config.gossip_rounds} rounds...")
        self.training = True
        self.round_count = 0

        while self.training and self.round_count < self.config.gossip_rounds:
            peer = self.network.get_random_peer()
            if not peer:
                print("⚠️ No peers available")
                time.sleep(self.config.gossip_interval)
                continue

            print(f"Round {self.round_count + 1}: Contacting {peer[0]}:{peer[1]}")

            message = {'type': 'gossip_request'}
            response = self.network.send_message(peer[0], peer[1], message)

            if response and response.get('parameters'):
                if self.model.update_parameters(response['parameters']):
                    print(f"✅ Updated enhanced parameters from {peer[0]}:{peer[1]}")
                else:
                    print(f"❌ Failed to update parameters")
            else:
                print(f"⚠️ No response from {peer[0]}:{peer[1]}")

            self.round_count += 1

            if self.round_count < self.config.gossip_rounds:
                time.sleep(self.config.gossip_interval)

        self.training = False
        print(f"🎉 Enhanced gossip training completed!")

    def test_predictions(self):
        """Test predictions with detailed output for new songs (no existing ratings)"""
        print("\n🔮 Testing enhanced predictions for NEW songs:")

        # Get all available users and songs
        all_users = list(self.model.user_ratings.keys())
        all_songs = list(self.model.song_attributes.keys())

        if not all_users or not all_songs:
            print("❌ No users or songs available for testing")
            return

        # Select a few users to test
        test_users = all_users[:3] if len(all_users) >= 3 else all_users
        predictions_made = 0
        target_predictions = 5

        print(f"Testing for users: {test_users}")
        print("-" * 60)

        for user_id in test_users:
            if predictions_made >= target_predictions:
                break

            # Get songs this user has NOT rated
            user_rated_songs = set(self.model.user_ratings.get(user_id, {}).keys())
            unrated_songs = [song_id for song_id in all_songs if song_id not in user_rated_songs]

            if not unrated_songs:
                print(f"⚠️ User {user_id} has rated all available songs")
                continue

            # Test a few random unrated songs for this user
            test_songs = random.sample(unrated_songs, min(3, len(unrated_songs)))

            for song_id in test_songs:
                if predictions_made >= target_predictions:
                    break

                pred = self.model.predict(user_id, song_id)
                song_info = self.model.get_song_info(song_id)

                # Get content similarity score for context
                content_sim = self.model._content_similarity(user_id, song_id)

                print(f"👤 User {user_id} → 🎵 {song_info}")
                print(f"   Predicted Rating: {pred:.2f} ⭐ (Content Similarity: {content_sim:.2f})")

                # Show user preferences if available
                if user_id in self.model.user_preferences:
                    prefs = self.model.user_preferences[user_id]
                    print(f"   User preferences: Energy={prefs.get('energy', 0):.2f}, "
                          f"Danceability={prefs.get('danceability', 0):.2f}, "
                          f"Genre={prefs.get('genre', 'Unknown')}")

                # Show song attributes for comparison
                if song_id in self.model.song_attributes:
                    attrs = self.model.song_attributes[song_id]
                    energy_norm = attrs.get('energy_normalized', attrs.get('energy', 0))
                    dance_norm = attrs.get('danceability_normalized', attrs.get('danceability', 0))
                    print(f"   Song attributes: Energy={energy_norm:.2f}, "
                          f"Danceability={dance_norm:.2f}, "
                          f"Genre={attrs.get('genre', 'Unknown')}")

                print("-" * 60)
                predictions_made += 1

        if predictions_made == 0:
            print("⚠️ No new songs found to test - all users have rated all available songs")
            print("💡 This suggests the recommendation system is working with limited data")
        else:
            print(f"✅ Generated {predictions_made} predictions for new song recommendations")


def main():
    """Main entry point"""
    print("🎵 Enhanced Synchronized Gossip Music Recommendation System")
    print("=" * 65)

    config = Config.load()
    print(f"Node: {config.host}:{config.port}")
    print(f"Peers: {config.peers}")
    print(f"Ratings file: {config.ratings_file}")
    print(f"Songs file: {config.songs_file}")

    node = EnhancedGossipNode(config)

    if not node.start():
        sys.exit(1)

    try:
        node.run_gossip()
        node.test_predictions()

        print("\nEnhanced node running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n⏹️ Stopping...")
        node.stop()


if __name__ == "__main__":
    main()