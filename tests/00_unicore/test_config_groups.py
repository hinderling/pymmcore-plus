import pytest

from pymmcore_plus.experimental.unicore import (
    StateDevice,
    UniMMCore,
)


def test_config_groups_with_python_state_device():
    core = UniMMCore()

    class SimStateDevice(StateDevice):
        def __init__(self, label: str, state_dict: dict[int, str]) -> None:
            super().__init__(state_dict)
            self._current_state = 0
            self._current_label = self._state_to_label.get(self._current_state)
            self._label = label

        def get_state(self) -> int:
            return self._current_state

        def set_state(self, position: int | str) -> None:
            if isinstance(position, str):
                position = int(position)
            self._current_state = position
            self._current_label = self._state_to_label.get(self._current_state)

    core.loadPyDevice(
        "PyLED", SimStateDevice(label="PyLED", state_dict={0: "UV", 1: "BLUE"})
    )
    core.initializeDevice("PyLED")

    # Define group and config using Python device
    core.defineConfigGroup("my_group")
    core.defineConfig("my_group", "uv_cfg", "PyLED", "Label", "UV")

    # Lists should include python-side groups/configs
    assert "my_group" in core.getAvailableConfigGroups()
    assert "uv_cfg" in core.getAvailableConfigs("my_group")

    # Change to BLUE, then apply config back to UV
    core.setStateLabel("PyLED", "BLUE")
    assert core.getStateLabel("PyLED") == "BLUE"
    core.setConfig("my_group", "uv_cfg")
    assert core.getStateLabel("PyLED") == "UV"

    # Group state fallback (python mapping)
    state = core.getConfigGroupState("my_group")
    assert isinstance(state, dict)
    assert state["PyLED"]["Label"] == "UV"

    # Cache-based state should include python values via Configuration
    cached_state = core.getConfigGroupStateFromCache("my_group")
    assert ("PyLED", "Label", "UV") in list(cached_state)

    # waiting on a python-only config should not error
    core.waitForConfig("my_group", "uv_cfg")

    # Native-only should fail for python-defined groups
    with pytest.raises(RuntimeError):
        core.getConfigGroupState("my_group", native=True)

    # Rename and delete configs in python store
    core.renameConfig("my_group", "uv_cfg", "uv_cfg2")
    assert "uv_cfg2" in core.getAvailableConfigs("my_group")
    assert "uv_cfg" not in core.getAvailableConfigs("my_group")

    core.deleteConfig("my_group", "uv_cfg2")
    assert "uv_cfg2" not in core.getAvailableConfigs("my_group")

    # Also support empty two-arg defineConfig
    core.defineConfig("empty_group", "empty")
    assert "empty_group" in core.getAvailableConfigGroups()


def test_config_group_introspection_with_python_device():
    core = UniMMCore()

    class SimStateDevice(StateDevice):
        def __init__(self, label: str, state_dict: dict[int, str]) -> None:
            super().__init__(state_dict)
            self._current_state = 0
            self._current_label = self._state_to_label.get(self._current_state)
            self._label = label

        def get_state(self) -> int:
            return self._current_state

        def set_state(self, position: int | str) -> None:
            if isinstance(position, str):
                position = int(position)
            self._current_state = position
            self._current_label = self._state_to_label.get(self._current_state)

    core.loadPyDevice(
        "PyLED", SimStateDevice(label="PyLED", state_dict={0: "UV", 1: "BLUE"})
    )
    core.initializeAllDevices()

    # Define two configs
    core.defineConfigGroup("grp")
    core.defineConfig("grp", "uv", "PyLED", "Label", "UV")
    core.defineConfig("grp", "blue", "PyLED", "Label", "BLUE")

    # getConfigData should return stored triplets
    data = core.getConfigData("grp", "uv")
    assert ("PyLED", "Label", "UV") in data

    # getConfigState should reflect stored values (not live), mapping-style
    state = core.getConfigState("grp", "uv")
    assert state["PyLED"]["Label"] == "UV"

    # Change current to BLUE; current config should be 'blue'
    core.setStateLabel("PyLED", "BLUE")
    assert core.getStateLabel("PyLED") == "BLUE"
    assert core.getCurrentConfig("grp") == "blue"

    # Switch back by applying 'uv', then verify current detection from cache
    core.setConfig("grp", "uv")
    assert core.getStateLabel("PyLED") == "UV"
    # Ensure cache path works too
    current_cached = core.getCurrentConfigFromCache("grp")
    assert current_cached == "uv"

    # System state should include python device properties
    system_state = core.getSystemState()
    assert ("PyLED", "Label", "UV") in list(system_state)
    cached_system_state = core.getSystemStateCache()
    assert ("PyLED", "Label", "UV") in list(cached_system_state)


def test_mixed_native_and_python_devices_in_one_config():
    core = UniMMCore()
    # Load native demo configuration to have C++ devices available
    core.loadSystemConfiguration()

    # Sanity check a native device is present (from demo config)
    assert "Camera" in core.getLoadedDevices()

    # Add a python StateDevice
    class SimStateDevice(StateDevice):
        def __init__(self, label: str, state_dict: dict[int, str]) -> None:
            super().__init__(state_dict)
            self._current_state = 0
            self._current_label = self._state_to_label.get(self._current_state)
            self._label = label

        def get_state(self) -> int:
            return self._current_state

        def set_state(self, position: int | str) -> None:
            if isinstance(position, str):
                position = int(position)
            self._current_state = position
            self._current_label = self._state_to_label.get(self._current_state)

    core.loadPyDevice(
        "PyLED", SimStateDevice(label="PyLED", state_dict={0: "UV", 1: "BLUE"})
    )
    core.initializeAllDevices()

    # Choose a native property to include in the config; use Binning on Camera
    # Ensure we can set it and read it
    native_dev = "Camera"
    native_prop = "Binning"
    # Some demo configs default to 1; set to 2 then back via config
    core.setProperty(native_dev, native_prop, "2")
    assert core.getProperty(native_dev, native_prop) == "2"

    # Define a mixed config: python LED Label=UV and native Camera Binning=1
    core.defineConfigGroup("mix")
    core.defineConfig("mix", "cfg", "PyLED", "Label", "UV")
    core.defineConfig("mix", "cfg", native_dev, native_prop, "1")

    # Change current away from the target config
    core.setStateLabel("PyLED", "BLUE")
    assert core.getStateLabel("PyLED") == "BLUE"
    core.setProperty(native_dev, native_prop, "2")
    assert core.getProperty(native_dev, native_prop) == "2"

    # Apply mixed config; both python and native properties should be applied
    core.setConfig("mix", "cfg")
    assert core.getStateLabel("PyLED") == "UV"
    assert core.getProperty(native_dev, native_prop) == "1"

    # Group state via python fallback should include LED Label=UV
    st = core.getConfigGroupState("mix")
    assert st["PyLED"]["Label"] == "UV"

    # Native-only query may succeed or fail depending on C++ group presence.
    # Ensure it doesn't break the API surface.
    # If it succeeds, it should return a native object; if it fails, it should raise.
    try:
        _ = core.getConfigGroupState("mix", native=True)
    except RuntimeError:
        pass


def test_rename_and_delete_python_config_group():
    core = UniMMCore()

    class SimStateDevice(StateDevice):
        def __init__(self, label: str, state_dict: dict[int, str]) -> None:
            super().__init__(state_dict)
            self._current_state = 0
            self._current_label = self._state_to_label.get(self._current_state)
            self._label = label

        def get_state(self) -> int:
            return self._current_state

        def set_state(self, position: int | str) -> None:
            if isinstance(position, str):
                position = int(position)
            self._current_state = position
            self._current_label = self._state_to_label.get(self._current_state)

    core.loadPyDevice(
        "PyLED", SimStateDevice(label="PyLED", state_dict={0: "UV", 1: "BLUE"})
    )
    core.initializeDevice("PyLED")

    core.defineConfigGroup("py_group")
    core.defineConfig("py_group", "uv", "PyLED", "Label", "UV")

    core.renameConfigGroup("py_group", "py_group_renamed")
    groups = core.getAvailableConfigGroups()
    assert "py_group" not in groups
    assert "py_group_renamed" in groups

    # Deleting should remove python configs as well
    core.deleteConfigGroup("py_group_renamed")
    assert "py_group_renamed" not in core.getAvailableConfigGroups()


def test_state_label_synchronization():
    """Verify State and Label stay synchronized during config operations."""
    core = UniMMCore()

    class SimStateDevice(StateDevice):
        def __init__(self, label: str, state_dict: dict[int, str]) -> None:
            super().__init__(state_dict)
            self._current_state = 0
            self._current_label = self._state_to_label.get(self._current_state)

        def get_state(self) -> int:
            return self._current_state

        def set_state(self, position: int | str) -> None:
            if isinstance(position, str):
                position = int(position)
            self._current_state = position
            self._current_label = self._state_to_label.get(self._current_state)

    core.loadPyDevice(
        "PyLED",
        SimStateDevice(label="PyLED", state_dict={0: "UV", 1: "BLUE", 2: "RED"}),
    )
    core.initializeDevice("PyLED")

    # Test 1: Config with State property should set Label too
    core.defineConfigGroup("test")
    core.defineConfig("test", "state_cfg", "PyLED", "State", "1")
    core.setConfig("test", "state_cfg")

    assert core.getState("PyLED") == 1
    assert core.getStateLabel("PyLED") == "BLUE"  # Should be synced!

    # Test 2: Config with Label property should set State too
    core.defineConfig("test", "label_cfg", "PyLED", "Label", "RED")
    core.setConfig("test", "label_cfg")

    assert core.getStateLabel("PyLED") == "RED"
    assert core.getState("PyLED") == 2  # Should be synced!

    # Test 3: Invalid state should raise error (not fall back silently)
    core.defineConfig("test", "invalid_cfg", "PyLED", "State", "999")
    with pytest.raises(ValueError, match="Position 999 is not a valid state"):
        core.setConfig("test", "invalid_cfg")

    # Test 4: Invalid label should raise error
    core.defineConfig("test", "invalid_label", "PyLED", "Label", "INVALID")
    with pytest.raises(RuntimeError, match="Label not defined: 'INVALID'"):
        core.setConfig("test", "invalid_label")


def test_multiple_properties_per_device_in_config():
    """Test configs with multiple properties for the same device."""
    core = UniMMCore()

    class ComplexStateDevice(StateDevice):
        def __init__(self, label: str, state_dict: dict[int, str]) -> None:
            super().__init__(state_dict)
            self._current_state = 0
            self._current_label = self._state_to_label.get(self._current_state)
            self._intensity = 50  # Additional property

        def get_state(self) -> int:
            return self._current_state

        def set_state(self, position: int | str) -> None:
            if isinstance(position, str):
                position = int(position)
            self._current_state = position
            self._current_label = self._state_to_label.get(self._current_state)

        def get_property_names(self) -> tuple[str, ...]:
            return (*super().get_property_names(), "Intensity")

        def get_property_value(self, prop_name: str) -> int | str:
            if prop_name == "Intensity":
                return self._intensity
            return super().get_property_value(prop_name)

        def set_property_value(self, prop_name: str, value: int | str) -> None:
            if prop_name == "Intensity":
                self._intensity = int(value)
            else:
                super().set_property_value(prop_name, value)

    core.loadPyDevice(
        "PyLED",
        ComplexStateDevice(label="PyLED", state_dict={0: "UV", 1: "BLUE"}),
    )
    core.initializeDevice("PyLED")

    # Define config with multiple properties for same device
    core.defineConfigGroup("channel")
    core.defineConfig("channel", "uv_bright", "PyLED", "Label", "UV")
    core.defineConfig("channel", "uv_bright", "PyLED", "Intensity", "100")

    # Set to different state first
    core.setStateLabel("PyLED", "BLUE")
    core.setProperty("PyLED", "Intensity", "50")

    # Apply config - should set both properties
    core.setConfig("channel", "uv_bright")

    assert core.getStateLabel("PyLED") == "UV"
    assert core.getProperty("PyLED", "Intensity") == 100  # Returns int, not string

    # Verify config data includes both properties
    config_data = core.getConfigData("channel", "uv_bright")
    assert ("PyLED", "Label", "UV") in config_data
    assert ("PyLED", "Intensity", "100") in config_data


def test_is_config_defined_for_pydevice():
    """Test isConfigDefined works for pydevice-only configs."""
    core = UniMMCore()

    class SimStateDevice(StateDevice):
        def __init__(self, label: str, state_dict: dict[int, str]) -> None:
            super().__init__(state_dict)
            self._current_state = 0
            self._current_label = self._state_to_label.get(self._current_state)

        def get_state(self) -> int:
            return self._current_state

        def set_state(self, position: int | str) -> None:
            if isinstance(position, str):
                position = int(position)
            self._current_state = position
            self._current_label = self._state_to_label.get(self._current_state)

    core.loadPyDevice(
        "PyLED", SimStateDevice(label="PyLED", state_dict={0: "UV", 1: "BLUE"})
    )
    core.initializeDevice("PyLED")

    # Before defining config
    assert not core.isConfigDefined("test", "cfg1")

    # After defining config
    core.defineConfigGroup("test")
    core.defineConfig("test", "cfg1", "PyLED", "Label", "UV")
    assert core.isConfigDefined("test", "cfg1")

    # Non-existent config
    assert not core.isConfigDefined("test", "cfg_doesnt_exist")

    # Non-existent group
    assert not core.isConfigDefined("group_doesnt_exist", "cfg1")
