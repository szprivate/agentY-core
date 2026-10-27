import unittest

from agenty_core import dashscope_endpoints as e

PLAN = "sk-sp-abc123"
PAY = "sk-abc123"


class Endpoints(unittest.TestCase):
    def test_prefix(self):
        self.assertTrue(e.is_plan_key(PLAN))
        self.assertFalse(e.is_plan_key(PAY))
        self.assertFalse(e.is_plan_key(""))

    def test_url_mapping(self):
        self.assertEqual(e.endpoint_for(e.PAYG, "singapore"), "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
        self.assertEqual(e.endpoint_for(e.PAYG, "beijing"), "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.assertEqual(e.endpoint_for(e.PAYG, "us"), "https://dashscope-us.aliyuncs.com/compatible-mode/v1")
        self.assertEqual(e.endpoint_for(e.TOKEN_PLAN, "singapore"),
                         "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1")
        self.assertEqual(e.endpoint_for(e.CODING_PLAN, "singapore"), "https://coding-intl.dashscope.aliyuncs.com/v1")
        with self.assertRaises(ValueError):
            e.endpoint_for(e.TOKEN_PLAN, "us")
        self.assertEqual(e.DEFAULT_BASE_URL, e.endpoint_for(e.PAYG, "singapore"))

    def test_plan_of_url(self):
        for (plan, _r), url in e.ENDPOINTS.items():
            self.assertEqual(e.plan_of_url(url), plan, url)
        self.assertEqual(e.plan_of_url("https://ws-1.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"), e.PAYG)
        self.assertIsNone(e.plan_of_url("https://example.com/v1"))

    def test_native_url(self):
        self.assertEqual(e.native_url(e.DEFAULT_BASE_URL), "https://dashscope-intl.aliyuncs.com/api/v1")
        self.assertEqual(e.native_url("https://ws-1.maas.aliyuncs.com/compatible-mode/v1"),
                         "https://ws-1.maas.aliyuncs.com/api/v1")
        self.assertIsNone(e.native_url(e.endpoint_for(e.TOKEN_PLAN, "beijing")))
        self.assertIsNone(e.native_url(e.endpoint_for(e.CODING_PLAN, "beijing")))
        self.assertIsNone(e.native_url(""))

    def test_candidates(self):
        self.assertEqual({p for p, _, _ in e.candidates_for_key(PLAN)}, {e.TOKEN_PLAN, e.CODING_PLAN})
        self.assertEqual({p for p, _, _ in e.candidates_for_key(PAY)}, {e.PAYG})

    def test_detect_uses_first_200(self):
        seen = []

        def fake(url, key):
            seen.append(url)
            return "ap-southeast-1" in url
        got = e.detect(PLAN, fake)
        self.assertEqual(got[:2], (e.TOKEN_PLAN, "singapore"))
        self.assertIsNone(e.detect(PAY, lambda u, k: False))


class Precedence(unittest.TestCase):
    def test_payg_key_wins(self):
        env = {"DASHSCOPE_API_KEY": PLAN, "DASHSCOPE_BASE_URL": e.endpoint_for(e.TOKEN_PLAN, "singapore"),
               "DASHSCOPE_PAYG_API_KEY": PAY, "DASHSCOPE_PAYG_BASE_URL": e.endpoint_for(e.PAYG, "beijing")}
        self.assertEqual(e.payg_credentials(env), (PAY, e.endpoint_for(e.PAYG, "beijing"), ""))
        self.assertEqual(e.chat_credentials(env)[0], PLAN)

    def test_payg_key_default_url(self):
        self.assertEqual(e.payg_credentials({"DASHSCOPE_PAYG_API_KEY": PAY})[1], e.DEFAULT_BASE_URL)

    def test_falls_back_to_main_payg_key(self):
        env = {"DASHSCOPE_API_KEY": PAY, "DASHSCOPE_BASE_URL": e.endpoint_for(e.PAYG, "us")}
        self.assertEqual(e.payg_credentials(env), (PAY, e.endpoint_for(e.PAYG, "us"), ""))

    def test_plan_only_names_the_fix(self):
        key, _base, err = e.payg_credentials({"DASHSCOPE_API_KEY": PLAN})
        self.assertEqual(key, "")
        self.assertIn("DASHSCOPE_PAYG_API_KEY", err)
        key, _b, err = e.payg_credentials({"DASHSCOPE_API_KEY": PAY,
                                           "DASHSCOPE_BASE_URL": e.endpoint_for(e.CODING_PLAN, "beijing")})
        self.assertEqual(key, "")
        self.assertIn("DASHSCOPE_PAYG_API_KEY", err)

    def test_no_key(self):
        self.assertTrue(e.payg_credentials({})[2])


if __name__ == "__main__":
    unittest.main()
