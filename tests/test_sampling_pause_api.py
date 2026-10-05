import asyncio
import pathlib
import sys
import types
import unittest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from test_sampler_pause import load
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

class ApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_routes_pending_resume_idle_queue_clear_shutdown(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        self.assertTrue((root / 'sampling_pause_api.py').exists(), 'Pause API not implemented')
        c = load('pause_controller').PauseController()
        server = types.SimpleNamespace(routes=web.RouteTableDef(), app=web.Application())
        @server.routes.post('/queue')
        async def queue(request):
            return web.json_response(await request.json())
        load('sampling_pause_api').register_routes(server, c)
        server.app.add_routes(server.routes)
        client = TestClient(TestServer(server.app))
        await client.start_server()
        try:
            response = await client.post('/sampling_pause')
            self.assertEqual(response.status, 409)
            c.begin('euler', True)
            response = await client.post('/sampling_pause')
            self.assertEqual(response.status, 200)
            self.assertTrue((await response.json())['pause_requested'])
            await client.post('/sampling_resume')
            self.assertFalse(c.status()['pause_requested'])
            c.request_pause()
            server.app.router  # normal HTTP remains available
            response = await client.get('/sampling_pause/status')
            self.assertTrue((await response.json())['sampling_active'])
            response = await client.post('/queue', json={'clear': True})
            self.assertEqual(await response.json(), {'clear': True})
            self.assertFalse(c.status()['pause_requested'])
            c.request_pause()
        finally:
            await client.close()
        self.assertFalse(c.status()['pause_requested'])

if __name__ == '__main__': unittest.main()
