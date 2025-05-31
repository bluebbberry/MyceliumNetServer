#!/usr/bin/env python3
import json
import sys
import time
import signal
from gossip import GossipLearning


class MusicGossipApp:
    def __init__(self):
        self.gossip_learner = None
        self.config = None
        self.load_config()

    def load_config(self):
        """Load configuration from config.json"""
        try:
            with open("config.json", 'r') as f:
                self.config = json.load(f)
            print(f"Loaded config: Node {self.config['host']}:{self.config['port']}")
        except FileNotFoundError:
            print("Error: config.json not found!")
            print("Please create a config.json file with your node configuration.")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading config: {e}")
            sys.exit(1)

    def setup_gossip_learner(self):
        """Initialize GOSSIP learner with config"""
        self.gossip_learner = GossipLearning(
            host=self.config["host"],
            port=self.config["port"],
            peers=[(peer[0], peer[1]) for peer in self.config["peers"]]
        )

    def run(self):
        """Main application run loop"""
        print("=" * 50)
        print("🎵 GOSSIP Music Recommendation System")
        print("=" * 50)

        # Setup
        self.setup_gossip_learner()

        # Start network
        print(f"Starting network on {self.config['host']}:{self.config['port']}...")
        self.gossip_learner.start_network()

        # Load data
        data_file = self.config.get("data_file", "ratings.csv")
        print(f"Loading training data from {data_file}...")

        if not self.gossip_learner.load_data(data_file):
            print(f"❌ Failed to load data from {data_file}")
            print("Make sure the file exists and has the correct format.")
            sys.exit(1)

        print("✅ Data loaded and local model trained!")

        # Start training if configured
        if self.config.get("auto_start_training", True):
            self.start_training()
        else:
            print("Auto-training disabled. Node is ready for manual commands.")
            self.keep_running()

    def start_training(self):
        """Start GOSSIP training"""
        rounds = self.config.get("gossip_rounds", 30)
        print(f"\n🔄 Starting GOSSIP training for {rounds} rounds...")
        print("Connecting to peers and exchanging model parameters...")

        # Set gossip interval if specified
        if "gossip_interval" in self.config:
            self.gossip_learner.gossip_interval = self.config["gossip_interval"]

        # Start training
        if not self.gossip_learner.start_gossip_training(rounds):
            print("❌ Failed to start GOSSIP training")
            return

        # Handle Ctrl+C gracefully
        def signal_handler(sig, frame):
            print("\n⏹️  Stopping training...")
            self.gossip_learner.stop_training()
            self.save_and_test()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        print("Training started! Press Ctrl+C to stop early.")
        print("=" * 50)

        # Wait for training to complete
        try:
            self.gossip_learner.wait_for_training_completion()
            print("\n✅ GOSSIP training completed!")
            self.save_and_test()

        except KeyboardInterrupt:
            print("\n⏹️  Training interrupted by user")
            self.gossip_learner.stop_training()
            self.save_and_test()

    def save_and_test(self):
        """Save model and run test predictions"""
        # Save model
        model_file = self.config.get("model_file", "model.pkl")
        print(f"\n💾 Saving model to {model_file}...")

        if self.gossip_learner.save_model(model_file):
            print("✅ Model saved successfully!")
        else:
            print("❌ Failed to save model")

        # Test predictions if configured
        if self.config.get("auto_predict_after_training", True):
            self.test_predictions()

    def test_predictions(self):
        """Run test predictions"""
        print("\n🔮 Testing predictions...")

        test_user_id = self.config.get("test_user_id", 1)
        num_recs = self.config.get("num_recommendations", 5)

        # Get recommendations
        recommendations = self.gossip_learner.recommend_songs(test_user_id, num_recs)

        if recommendations:
            print(f"\n🎵 Top {len(recommendations)} recommendations for user {test_user_id}:")
            print("-" * 40)
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. Song {rec['song_id']:>3}: {rec['predicted_rating']:.2f} ⭐")
        else:
            print(f"❌ No recommendations available for user {test_user_id}")

        # Test specific prediction
        print(f"\n🎯 Testing specific prediction:")
        rating = self.gossip_learner.predict_rating(test_user_id, 101)
        if rating is not None:
            print(f"User {test_user_id} + Song 101 = {rating:.2f} ⭐")
        else:
            print("Could not predict rating for user 1, song 101")

    def keep_running(self):
        """Keep the node running for manual interaction"""
        try:
            print("\nNode is running. Press Ctrl+C to stop.")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n👋 Shutting down...")

        finally:
            if self.gossip_learner:
                self.gossip_learner.stop_network()


def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        print("This application uses config.json for all settings.")
        print("Simply run: python main.py")
        print("\nTo configure different nodes, edit config.json:")
        print("- Change 'port' for each node")
        print("- Update 'peers' list accordingly")
        sys.exit(1)

    app = MusicGossipApp()
    app.run()


if __name__ == "__main__":
    main()