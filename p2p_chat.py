import asyncio
import sys

class Peer:
    def __init__(self, port):
        self.port = port
        self.peers = set()

    async def handle_connection(self, reader, writer):
        peer_addr = writer.get_extra_info('peername')
        print(f"[+] Incoming connection from {peer_addr}")
        self.peers.add((reader, writer))
        await self.receive_messages(reader)

    async def receive_messages(self, reader):
        try:
            while True:
                data = await reader.readline()
                if not data:
                    break
                message = data.decode().strip()
                print(f"[<] {message}")
        except ConnectionResetError:
            pass

    async def start_server(self):
        server = await asyncio.start_server(
            self.handle_connection, '127.0.0.1', self.port
        )
        print(f"[i] Listening on 127.0.0.1:{self.port}")
        async with server:
            await server.serve_forever()

    async def connect_to_peer(self, host, port):
        try:
            reader, writer = await asyncio.open_connection(host, port)
            self.peers.add((reader, writer))
            print(f"[+] Connected to peer at {host}:{port}")
            asyncio.create_task(self.receive_messages(reader))
        except Exception as e:
            print(f"[!] Failed to connect: {e}")

    async def send_loop(self):
        loop = asyncio.get_event_loop()
        while True:
            message = await loop.run_in_executor(None, sys.stdin.readline)
            if message.strip().startswith("/connect"):
                _, host, port = message.strip().split()
                await self.connect_to_peer(host, int(port))
            else:
                for _, writer in list(self.peers):
                    writer.write(message.encode())
                    await writer.drain()

async def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9000
    peer = Peer(port)

    await asyncio.gather(
        peer.start_server(),
        peer.send_loop()
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[i] Exiting.")
