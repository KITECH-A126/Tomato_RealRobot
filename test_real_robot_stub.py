"""Offline tests: SDK construction is mocked before RealRobot is instantiated."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from real_robot_stub import RealRobot


class RealRobotTests(unittest.TestCase):
    def setUp(self):
        self.drivers = []
        for count, stamp, version in ((7, 100.0, "1.07"), (6, 101.0, "S-V1.8-2")):
            driver = Mock()
            driver.joint_nums = count
            driver.get_fps.return_value = 200.0
            driver.get_firmware.return_value = {"software_version": version}
            driver.get_joint_angles.return_value = SimpleNamespace(
                msg=[(-1) ** i * (i + 1) / 10 for i in range(count)],
                timestamp=stamp, hz=100.0)
            driver.is_connected.return_value = True
            driver.get_auto_set_motion_mode_enabled.return_value = True
            driver.get_joint_limits_enabled.return_value = False
            self.drivers.append(driver)
        self.factory = patch("pyAgxArm.AgxArmFactory.create_arm", side_effect=self.drivers)
        self.factory.start()
        self.addCleanup(self.factory.stop)

    def bot(self):
        bot = RealRobot()
        self.addCleanup(bot.close)
        return bot

    def test_read_preserves_order_values_and_cached_times(self):
        bot = self.bot()
        state = bot.read_joints()
        self.assertEqual(state["nero"], self.drivers[0].get_joint_angles().msg)
        self.assertEqual(state["piper"], self.drivers[1].get_joint_angles().msg)
        self.assertEqual((state["grip"], state["t"]), (0.0, 100.0))
        self.assertEqual(bot.read_joints()["t"], state["t"])
        state["nero"][0] = 99
        self.assertEqual(bot.read_joints()["nero"][0], 0.1)
        for driver in self.drivers:
            driver.enable.assert_not_called()
            driver.move_js.assert_not_called()

    def test_command_once_each_without_modification(self):
        bot = self.bot()
        cmd = bot.read_joints()
        bot.command_joints(cmd)
        for arm, driver in zip(("nero", "piper"), self.drivers):
            driver.move_js.assert_called_once_with(cmd[arm])
            driver.move_j.assert_not_called()

    def test_bad_second_arm_sends_neither(self):
        bot = self.bot()
        for values in ([0] * 5, [7.0] * 6, [float("nan")] * 6,
                       [float("inf")] * 6, [True] * 6, ["0"] * 6):
            cmd = bot.read_joints()
            cmd["piper"] = values
            with self.subTest(values=values), self.assertRaises(ValueError):
                bot.command_joints(cmd)
        for driver in self.drivers:
            driver.move_js.assert_not_called()

    def test_model_limits_checked_if_enabled(self):
        bot = self.bot()
        self.drivers[1].get_joint_limits_enabled.return_value = True
        cmd = bot.read_joints()
        cmd["piper"] = [0.0, -0.1, 0.0, 0.0, 0.0, 0.0]
        with self.assertRaises(ValueError):
            bot.command_joints(cmd)
        self.drivers[0].move_js.assert_not_called()

    def test_missing_feedback_does_not_fabricate_state(self):
        bot = self.bot()
        self.drivers[1].get_joint_angles.return_value = None
        with self.assertRaises(RuntimeError):
            bot.read_joints()

    def test_close_twice_and_calls_after_close(self):
        bot = self.bot()
        cmd = bot.read_joints()
        bot.close()
        bot.close()
        for driver in self.drivers:
            driver.disconnect.assert_called_once()
        with self.assertRaises(RuntimeError):
            bot.read_joints()
        with self.assertRaises(RuntimeError):
            bot.command_joints(cmd)

    def test_connection_failure_cleans_up_both(self):
        self.drivers[1].connect.side_effect = RuntimeError("CAN unavailable")
        with self.assertRaisesRegex(RuntimeError, "CAN unavailable"):
            RealRobot()
        for driver in self.drivers:
            driver.disconnect.assert_called_once()

    def test_new_firmware_recreates_driver(self):
        self.drivers[0].get_firmware.return_value = {"software_version": "1.20"}
        replacement = Mock(wraps=None)
        replacement.joint_nums = 7
        replacement.get_joint_angles.return_value = self.drivers[0].get_joint_angles.return_value
        with patch("pyAgxArm.AgxArmFactory.create_arm",
                   side_effect=[self.drivers[0], replacement, self.drivers[1]]) as factory:
            self.bot()
            self.assertEqual(factory.call_args_list[1].args[0]["firmeware_version"], "v120")
            self.drivers[0].disconnect.assert_called_once()
            replacement.connect.assert_called_once()

    def test_waits_for_receive_ready_before_firmware_query(self):
        driver = self.drivers[1]
        driver.get_fps.side_effect = [0.0, 0.0, 200.0]
        events = []
        driver.get_firmware.side_effect = lambda **kw: events.append("firmware") or {
            "software_version": "S-V1.8-2"}
        with patch("real_robot_stub.time.sleep", side_effect=lambda _: events.append("wait")):
            self.bot()
        self.assertEqual(events, ["wait", "wait", "firmware"])

    def test_waits_for_first_joint_feedback(self):
        driver = self.drivers[0]
        valid = driver.get_joint_angles.return_value
        driver.get_joint_angles.side_effect = [None, valid]
        with patch("real_robot_stub.time.sleep") as sleep:
            self.bot()
        sleep.assert_called_once_with(0.01)

    def test_receive_timeout_closes_connection_without_firmware_request(self):
        self.drivers[0].get_fps.return_value = 0.0
        with patch("real_robot_stub.time.monotonic", side_effect=[0.0, 3.0]):
            with self.assertRaisesRegex(RuntimeError, "CAN 수신"):
                RealRobot()
        self.drivers[0].get_firmware.assert_not_called()
        self.drivers[0].disconnect.assert_called_once()

    def test_firmware_timeout_does_not_guess_profile(self):
        self.drivers[1].get_firmware.return_value = None
        with self.assertRaisesRegex(RuntimeError, "완전한 펌웨어 응답"):
            RealRobot()
        for driver in self.drivers:
            driver.disconnect.assert_called_once()

    def test_no_gripper_or_camera_success_is_faked(self):
        bot = self.bot()
        for method, args in ((bot.set_gripper, (0.0,)), (bot.grab_wrist, ()), (bot.grab_gaze, ())):
            with self.assertRaises(NotImplementedError):
                method(*args)
        cmd = bot.read_joints()
        cmd["grip"] = 1.0
        with self.assertRaises(NotImplementedError):
            bot.command_joints(cmd)
        self.drivers[0].move_js.assert_not_called()

    def test_transport_failure_is_not_retried(self):
        bot = self.bot()
        self.drivers[1].move_js.side_effect = RuntimeError("send failed")
        with self.assertRaisesRegex(RuntimeError, "send failed"):
            bot.command_joints(bot.read_joints())
        for driver in self.drivers:
            driver.move_js.assert_called_once()


if __name__ == "__main__":
    unittest.main()
