"""Test actual conversion against HA's current tool-result interface, offline."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import unittest


class ToolResult:
    def __init__(self, name, result):
        self.tool_name = name
        self.tool_call_id = 'call-1'
        self.result = SimpleNamespace(data=result, error=False)

    @property
    def tool_result(self):
        raise AssertionError('Deprecated HA tool_result property accessed')


class ConversionTests(unittest.TestCase):
    def convert(self, result):
        path = Path(__file__).resolve().parents[1] / 'custom_components/azure_openai_conversation/conversation.py'
        tree = ast.parse(path.read_text())
        method = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_convert_content_to_param')
        module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), method], type_ignores=[])
        ast.fix_missing_locations(module)
        namespace = {'conversation': SimpleNamespace(ToolResultContent=ToolResult), 'FunctionCallOutput': dict, 'json_dumps': json.dumps}
        exec(compile(module, str(path), 'exec'), namespace)
        return namespace['_convert_content_to_param']([ToolResult('HassGetState', result)])

    def test_result_payload_is_preserved_without_deprecated_property(self):
        result = {'state': 'on', 'success': True}
        converted = self.convert(result)
        self.assertEqual(converted[0]['call_id'], 'call-1')
        self.assertEqual(json.loads(converted[0]['output']), result)


if __name__ == '__main__':
    unittest.main()
