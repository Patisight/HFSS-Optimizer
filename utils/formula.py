"""
公式解析器和计算器

支持以下公式语法:
- S(1,1), S(2,1) 等 - 获取 S 参数复数值
- Z(1,1), Z(2,1) 等 - 获取 Z 参数复数值
- dB(S(1,1)) - S 参数的 dB 值 (20*log10(|S|))
- mag(S(1,1)) - S 参数的幅值
- phase(S(1,1)) - S 参数的相位 (度)
- re(S(1,1)) - S 参数的实部
- im(S(1,1)) - S 参数的虚部
- min(dB(S(1,1))) - 频段内最小值
- max(dB(S(1,1))) - 频段内最大值
- mean(dB(S(1,1))) - 频段内平均值
- +, -, *, / - 算术运算
- 括号 - 优先级控制

示例公式:
- "dB(S(1,1))"
- "dB(S(1,1)) + dB(S(2,1))"
- "min(dB(S(1,1)))"
- "(dB(S(1,1)) + dB(S(2,1))) / 2"
- "max(dB(S(1,1))) - min(dB(S(1,1)))"  - 带宽计算
- "mag(Z(1,1))"
- "re(Z(2,1))"
"""

import re
import numpy as np
from typing import Tuple, Optional, List, Dict, Any, Union
from dataclasses import dataclass
from enum import Enum, auto


class TokenType(Enum):
    """词法单元类型"""
    NUMBER = auto()
    IDENTIFIER = auto()  # 函数名或标识符
    S_PARAM = auto()    # S(数字,数字)
    Z_PARAM = auto()    # Z(数字,数字)
    LPAREN = auto()
    RPAREN = auto()
    PLUS = auto()
    MINUS = auto()
    MULTIPLY = auto()
    DIVIDE = auto()
    COMMA = auto()
    EOF = auto()


@dataclass
class Token:
    """词法单元"""
    type: TokenType
    value: Any
    position: int  # 在原始字符串中的位置


class FormulaSyntaxError(Exception):
    """公式语法错误"""
    pass


class FormulaEvaluationError(Exception):
    """公式计算错误"""
    pass


class Tokenizer:
    """词法分析器 - 将公式字符串转换为词法单元序列"""
    
    TOKEN_REGEX = [
        (r'\d+\.?\d*', TokenType.NUMBER),  # 数字
        (r'S\s*\(\s*\d+\s*,\s*\d+\s*\)', TokenType.S_PARAM),  # S(数字,数字) - 必须放在 IDENTIFIER 之前
        (r'Z\s*\(\s*\d+\s*,\s*\d+\s*\)', TokenType.Z_PARAM),  # Z(数字,数字) - 必须放在 IDENTIFIER 之前
        (r'[a-zA-Z_][a-zA-Z0-9_]*', TokenType.IDENTIFIER),  # 标识符
        (r'\(', TokenType.LPAREN),
        (r'\)', TokenType.RPAREN),
        (r'\+', TokenType.PLUS),
        (r'-', TokenType.MINUS),
        (r'\*', TokenType.MULTIPLY),
        (r'/', TokenType.DIVIDE),
        (r',', TokenType.COMMA),
    ]
    
    def __init__(self, formula: str):
        self.formula = formula
        self.pos = 0
        self.tokens: List[Token] = []
    
    def tokenize(self) -> List[Token]:
        """将公式字符串转换为词法单元序列"""
        while self.pos < len(self.formula):
            # 跳过空白字符
            if self.formula[self.pos].isspace():
                self.pos += 1
                continue
            
            matched = False
            for pattern, token_type in self.TOKEN_REGEX:
                regex = re.compile(pattern)
                match = regex.match(self.formula, self.pos)
                if match:
                    value = match.group()
                    
                    # S 参数和 Z 参数特殊处理：去除空格
                    if token_type in (TokenType.S_PARAM, TokenType.Z_PARAM):
                        value = re.sub(r'\s+', '', value)
                    
                    self.tokens.append(Token(token_type, value, self.pos))
                    self.pos = match.end()
                    matched = True
                    break
            
            if not matched:
                raise FormulaSyntaxError(f"无法识别的字符 '{self.formula[self.pos]}' at position {self.pos}")
        
        self.tokens.append(Token(TokenType.EOF, None, self.pos))
        return self.tokens


class ASTNode:
    """抽象语法树节点基类"""
    pass


@dataclass
class NumberNode(ASTNode):
    """数字节点"""
    value: float


@dataclass
class SParamNode(ASTNode):
    """S 参数节点 - S(行, 列)"""
    row: int
    col: int


@dataclass
class ZParamNode(ASTNode):
    """Z 参数节点 - Z(行, 列)"""
    row: int
    col: int


@dataclass
class FunctionCallNode(ASTNode):
    """函数调用节点 - dB(x), min(x), max(x), mean(x)"""
    name: str
    arg: ASTNode


@dataclass
class BinaryOpNode(ASTNode):
    """二元运算节点 - a + b, a - b, a * b, a / b"""
    left: ASTNode
    operator: str
    right: ASTNode


class Parser:
    """递归下降解析器 - 将词法单元序列转换为抽象语法树
    
    语法规则:
    expr ::= term (('+' | '-') term)*
    term ::= factor (('*' | '/') factor)*
    factor ::= NUMBER | S_PARAM | FUNCTION '(' expr ')' | '(' expr ')'
    """
    
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
    
    def current_token(self) -> Token:
        """获取当前词法单元"""
        return self.tokens[self.pos]
    
    def eat(self, token_type: TokenType):
        """消耗指定类型的词法单元"""
        if self.current_token().type == token_type:
            self.pos += 1
        else:
            raise FormulaSyntaxError(
                f"期望 {token_type}, 实际 {self.current_token().type} "
                f"at position {self.current_token().position}"
            )
    
    def parse(self) -> ASTNode:
        """解析表达式"""
        node = self.expr()
        if self.current_token().type != TokenType.EOF:
            raise FormulaSyntaxError(
                f"解析后仍有剩余内容 at position {self.current_token().position}"
            )
        return node
    
    def expr(self) -> ASTNode:
        """解析加减运算"""
        node = self.term()
        
        while self.current_token().type in (TokenType.PLUS, TokenType.MINUS):
            op = self.current_token()
            self.eat(op.type)
            right = self.term()
            node = BinaryOpNode(node, op.value, right)
        
        return node
    
    def term(self) -> ASTNode:
        """解析乘除运算"""
        node = self.factor()
        
        while self.current_token().type in (TokenType.MULTIPLY, TokenType.DIVIDE):
            op = self.current_token()
            self.eat(op.type)
            right = self.factor()
            node = BinaryOpNode(node, op.value, right)
        
        return node
    
    def factor(self) -> ASTNode:
        """解析因子"""
        token = self.current_token()
        
        if token.type == TokenType.NUMBER:
            self.eat(TokenType.NUMBER)
            return NumberNode(float(token.value))
        
        elif token.type == TokenType.S_PARAM:
            self.eat(TokenType.S_PARAM)
            # 解析 S(行,列)
            match = re.match(r'S\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)', token.value)
            if not match:
                raise FormulaSyntaxError(f"无效的 S 参数格式: {token.value}")
            row, col = int(match.group(1)), int(match.group(2))
            return SParamNode(row, col)
        
        elif token.type == TokenType.Z_PARAM:
            self.eat(TokenType.Z_PARAM)
            # 解析 Z(行,列)
            match = re.match(r'Z\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)', token.value)
            if not match:
                raise FormulaSyntaxError(f"无效的 Z 参数格式: {token.value}")
            row, col = int(match.group(1)), int(match.group(2))
            return ZParamNode(row, col)
        
        elif token.type == TokenType.IDENTIFIER:
            self.eat(TokenType.IDENTIFIER)
            name = token.value
            
            # 检查是否是函数调用
            if self.current_token().type == TokenType.LPAREN:
                self.eat(TokenType.LPAREN)
                arg = self.expr()
                self.eat(TokenType.RPAREN)
                
                # 验证函数名
                valid_functions = ['dB', 'mag', 'phase', 're', 'im', 'min', 'max', 'mean', 'abs']
                if name not in valid_functions:
                    raise FormulaSyntaxError(f"未知的函数: {name}")
                
                return FunctionCallNode(name, arg)
            else:
                raise FormulaSyntaxError(f"标识符后缺少括号 at position {token.position}")
        
        elif token.type == TokenType.LPAREN:
            self.eat(TokenType.LPAREN)
            node = self.expr()
            self.eat(TokenType.RPAREN)
            return node
        
        else:
            raise FormulaSyntaxError(
                f"无法解析的词法单元 {token.type} at position {token.position}"
            )


class FormulaValidator:
    """公式验证器"""
    
    VALID_FUNCTIONS = ['dB', 'mag', 'phase', 're', 'im', 'min', 'max', 'mean', 'abs']
    VALID_IDENTIFIERS = VALID_FUNCTIONS + ['S', 'Z']  # S 和 Z 是参数的关键字
    
    def __init__(self, formula: str):
        self.formula = formula
        self.errors: List[str] = []
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        验证公式语法
        
        Returns:
            (是否有效, 错误列表)
        """
        self.errors = []
        
        if not self.formula or not self.formula.strip():
            self.errors.append("公式不能为空")
            return False, self.errors
        
        try:
            tokenizer = Tokenizer(self.formula)
            tokens = tokenizer.tokenize()
            parser = Parser(tokens)
            ast = parser.parse()
            
            # 额外的语义检查
            self._check_ast(ast)
            
        except FormulaSyntaxError as e:
            self.errors.append(str(e))
        except Exception as e:
            self.errors.append(f"未知错误: {e}")
        
        return len(self.errors) == 0, self.errors
    
    def _check_ast(self, node: ASTNode):
        """检查抽象语法树的语义"""
        if isinstance(node, NumberNode):
            pass
        
        elif isinstance(node, SParamNode):
            if node.row < 1 or node.col < 1:
                self.errors.append(f"S 参数行号和列号必须 >= 1: S({node.row},{node.col})")
        
        elif isinstance(node, ZParamNode):
            if node.row < 1 or node.col < 1:
                self.errors.append(f"Z 参数行号和列号必须 >= 1: Z({node.row},{node.col})")
        
        elif isinstance(node, FunctionCallNode):
            # 聚合函数只能有一个参数
            if node.name in ['min', 'max', 'mean']:
                if not isinstance(node.arg, (SParamNode, ZParamNode, FunctionCallNode, BinaryOpNode)):
                    self.errors.append(f"{node.name}() 函数的参数必须是 S 参数、Z 参数、dB() 表达式或算术表达式")
        
        elif isinstance(node, BinaryOpNode):
            self._check_ast(node.left)
            self._check_ast(node.right)
    
    @staticmethod
    def suggest_correction(formula: str) -> Optional[str]:
        """
        尝试修正常见错误
        
        当前支持的修正:
        - s11 -> S(1,1)
        - s21 -> S(2,1)
        - db -> dB
        """
        formula = formula.strip()
        
        # 常见拼写错误修正
        corrections = {
            r'\bs11\b': 'S(1,1)',
            r'\bs21\b': 'S(2,1)',
            r'\bs22\b': 'S(2,2)',
            r'\bs12\b': 'S(1,2)',
            r'\bdb\b': 'dB',
            r'\bmag\b': 'mag',
            r'\bmin\b': 'min',
            r'\bmax\b': 'max',
        }
        
        for pattern, replacement in corrections.items():
            new_formula = re.sub(pattern, replacement, formula, flags=re.IGNORECASE)
            if new_formula != formula:
                return new_formula
        
        return None


class SParameterData:
    """S 参数和 Z 参数数据容器"""
    
    def __init__(self):
        self.data: Dict[Tuple[int, int], Dict[str, np.ndarray]] = {}
        self.z_data: Dict[Tuple[int, int], Dict[str, np.ndarray]] = {}
        self.freq: Optional[np.ndarray] = None
    
    def set_s_param(self, row: int, col: int, real, imag):
        """设置 S 参数数据"""
        key = (row, col)
        # 防御性处理：确保 real 和 imag 不是 None
        if real is None or imag is None:
            raise ValueError(f"S({row},{col}) real or imag is None")
        self.data[key] = {
            'real': np.array(real),
            'imag': np.array(imag),
        }
    
    def set_z_param(self, row: int, col: int, real, imag):
        """设置 Z 参数数据"""
        key = (row, col)
        # 防御性处理：确保 real 和 imag 不是 None
        if real is None or imag is None:
            raise ValueError(f"Z({row},{col}) real or imag is None")
        self.z_data[key] = {
            'real': np.array(real),
            'imag': np.array(imag),
        }
    
    def get_complex_z(self, row: int, col: int) -> Optional[np.ndarray]:
        """获取复数形式的 Z 参数"""
        key = (row, col)
        if key not in self.z_data:
            return None
        
        real = self.z_data[key]['real']
        imag = self.z_data[key]['imag']
        return real + 1j * imag
    
    def get_z_real(self, row: int, col: int) -> Optional[np.ndarray]:
        """获取 Z 参数的实部"""
        key = (row, col)
        if key not in self.z_data:
            return None
        return self.z_data[key]['real']
    
    def get_z_imag(self, row: int, col: int) -> Optional[np.ndarray]:
        """获取 Z 参数的虚部"""
        key = (row, col)
        if key not in self.z_data:
            return None
        return self.z_data[key]['imag']
    
    def set_frequency(self, freq: np.ndarray):
        """设置频率数据"""
        self.freq = np.array(freq)
    
    def get_complex(self, row: int, col: int) -> Optional[np.ndarray]:
        """获取复数形式的 S 参数"""
        key = (row, col)
        if key not in self.data:
            return None
        
        real = self.data[key]['real']
        imag = self.data[key]['imag']
        return real + 1j * imag
    
    def get_magnitude(self, row: int, col: int) -> Optional[np.ndarray]:
        """获取幅值"""
        complex_val = self.get_complex(row, col)
        return np.abs(complex_val) if complex_val is not None else None
    
    def get_phase(self, row: int, col: int) -> Optional[np.ndarray]:
        """获取相位（度）"""
        complex_val = self.get_complex(row, col)
        return np.angle(complex_val, deg=True) if complex_val is not None else None
    
    def get_db(self, row: int, col: int) -> Optional[np.ndarray]:
        """获取 dB 值: 20 * log10(|S|)"""
        mag = self.get_magnitude(row, col)
        return 20 * np.log10(mag) if mag is not None else None
    
    def get_real(self, row: int, col: int) -> Optional[np.ndarray]:
        """获取实部"""
        key = (row, col)
        if key not in self.data:
            return None
        return self.data[key]['real']
    
    def get_imag(self, row: int, col: int) -> Optional[np.ndarray]:
        """获取虚部"""
        key = (row, col)
        if key not in self.data:
            return None
        return self.data[key]['imag']


class FormulaEvaluator:
    """公式计算器 - 使用实际数据计算公式结果"""
    
    def __init__(self, s_param_data: SParameterData):
        self.s_data = s_param_data
    
    def evaluate(self, formula: str) -> Tuple[float, float]:
        """
        计算公式结果
        
        Returns:
            (最终结果, actual值)
            - 如果公式包含聚合函数(min/max/mean)，返回聚合后的标量
            - 如果公式不含聚合函数，返回频段内所有点的值
        """
        # 先验证
        valid, errors = FormulaValidator(formula).validate()
        if not valid:
            raise FormulaEvaluationError(f"公式错误: {errors[0]}")
        
        # 解析
        tokenizer = Tokenizer(formula)
        tokens = tokenizer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        
        # 计算
        result = self._eval_node(ast)
        
        return result
    
    def _eval_node(self, node: ASTNode) -> Union[float, np.ndarray]:
        """递归计算 AST 节点"""
        if isinstance(node, NumberNode):
            return node.value
        
        elif isinstance(node, SParamNode):
            # 返回复数数组（未聚合）
            complex_val = self.s_data.get_complex(node.row, node.col)
            if complex_val is None:
                raise FormulaEvaluationError(f"S({node.row},{node.col}) 数据不存在")
            return complex_val
        
        elif isinstance(node, ZParamNode):
            # 返回复数数组（未聚合）
            complex_val = self.s_data.get_complex_z(node.row, node.col)
            if complex_val is None:
                raise FormulaEvaluationError(f"Z({node.row},{node.col}) 数据不存在")
            return complex_val
        
        elif isinstance(node, FunctionCallNode):
            return self._eval_function(node.name, node.arg)
        
        elif isinstance(node, BinaryOpNode):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            
            if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
                # 至少有一个是数组，进行逐元素运算
                if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
                    if left.shape != right.shape:
                        raise FormulaEvaluationError("数组形状不匹配")
                    if node.operator == '+':
                        return left + right
                    elif node.operator == '-':
                        return left - right
                    elif node.operator == '*':
                        return left * right
                    elif node.operator == '/':
                        return left / right
                elif isinstance(left, np.ndarray):
                    # 左数组右标量
                    if node.operator == '+':
                        return left + right
                    elif node.operator == '-':
                        return left - right
                    elif node.operator == '*':
                        return left * right
                    elif node.operator == '/':
                        return left / right
                else:
                    # 左标量右数组
                    if node.operator == '+':
                        return left + right
                    elif node.operator == '-':
                        return left - right
                    elif node.operator == '*':
                        return left * right
                    elif node.operator == '/':
                        return left / right
            else:
                # 都是标量
                if node.operator == '+':
                    return left + right
                elif node.operator == '-':
                    return left - right
                elif node.operator == '*':
                    return left * right
                elif node.operator == '/':
                    return left / right
        
        raise FormulaEvaluationError(f"无法处理的节点类型: {type(node)}")
    
    def _eval_function(self, name: str, arg: ASTNode) -> float:
        """计算函数调用"""
        arg_value = self._eval_node(arg)
        
        if name == 'dB':
            if isinstance(arg_value, np.ndarray):
                return 20 * np.log10(np.abs(arg_value))
            else:
                return 20 * np.log10(np.abs(arg_value))
        
        elif name == 'mag':
            if isinstance(arg_value, np.ndarray):
                return np.abs(arg_value)
            else:
                return np.abs(arg_value)
        
        elif name == 'phase':
            if isinstance(arg_value, np.ndarray):
                return np.angle(arg_value, deg=True)
            else:
                return np.angle(arg_value, deg=True)
        
        elif name == 're':
            if isinstance(arg_value, np.ndarray):
                return np.real(arg_value)
            else:
                return np.real(arg_value)
        
        elif name == 'im':
            if isinstance(arg_value, np.ndarray):
                return np.imag(arg_value)
            else:
                return np.imag(arg_value)
        
        elif name == 'abs':
            if isinstance(arg_value, np.ndarray):
                return np.abs(arg_value)
            else:
                return np.abs(arg_value)
        
        elif name == 'min':
            if not isinstance(arg_value, np.ndarray):
                return float(arg_value)
            result = np.min(arg_value)
            return float(result)
        
        elif name == 'max':
            if not isinstance(arg_value, np.ndarray):
                return float(arg_value)
            result = np.max(arg_value)
            return float(result)
        
        elif name == 'mean':
            if not isinstance(arg_value, np.ndarray):
                return float(arg_value)
            result = np.mean(arg_value)
            return float(result)
        
        raise FormulaEvaluationError(f"未知的函数: {name}")


def parse_formula(formula: str) -> Tuple[bool, Optional[ASTNode], Optional[str]]:
    """
    解析公式的便捷函数
    
    Returns:
        (是否成功, AST节点或None, 错误信息或None)
    """
    try:
        tokenizer = Tokenizer(formula)
        tokens = tokenizer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        return True, ast, None
    except Exception as e:
        return False, None, str(e)


def evaluate_formula(formula: str, s_param_data: SParameterData) -> Tuple[float, Optional[str]]:
    """
    计算公式的便捷函数
    
    Returns:
        (计算结果或np.ndarray, 错误信息或None)
    """
    try:
        evaluator = FormulaEvaluator(s_param_data)
        result = evaluator.evaluate(formula)
        return result, None
    except Exception as e:
        return None, str(e)


# 测试代码
if __name__ == "__main__":
    # 测试词法分析器
    test_formulas = [
        "dB(S(1,1))",
        "dB(S(1,1)) + dB(S(2,1))",
        "(dB(S(1,1)) + dB(S(2,1))) / 2",
        "min(dB(S(1,1)))",
        "max(dB(S(1,1))) - min(dB(S(1,1)))",
    ]
    
    logger.info(f"=" * 60)
    logger.info(f"公式解析测试")
    logger.info(f"=" * 60)
    
    for formula in test_formulas:
        logger.info(f"\n公式: {formula}")
        valid, errors = FormulaValidator(formula).validate()
        if valid:
            logger.info(f"  ✓ 语法正确")
            success, ast, err = parse_formula(formula)
            if success:
                logger.info(f"  AST: {ast}")
        else:
            logger.info(f"  ✗ 语法错误: {errors}")
    
    # 测试修正建议
    logger.info(f"\n" + "=" * 60)
    logger.info(f"公式修正建议测试")
    logger.info(f"=" * 60)
    
    misspellings = ["s11", "s21", "db(S(1,1))", "dB(s11) + dB(s21)"]
    for wrong in misspellings:
        suggestion = FormulaValidator.suggest_correction(wrong)
        if suggestion:
            logger.info(f"  '{wrong}' -> '{suggestion}'")
        else:
            logger.info(f"  '{wrong}' -> 无建议")
    
    # 测试计算器（模拟数据）
    logger.info(f"\n" + "=" * 60)
    logger.info(f"公式计算测试（模拟数据）")
    logger.info(f"=" * 60)
    
    # 创建模拟 S 参数数据
    s_data = SParameterData()
    s_data.set_frequency(np.linspace(5.6, 6.2, 100))
    
    # 设置 S11 模拟数据（简单的线性变化）
    real11 = np.linspace(-0.5, -0.1, 100)
    imag11 = np.linspace(0.1, 0.3, 100)
    s_data.set_s_param(1, 1, real11, imag11)
    
    # 设置 S21 模拟数据
    real21 = np.linspace(-0.3, -0.2, 100)
    imag21 = np.linspace(0.05, 0.1, 100)
    s_data.set_s_param(2, 1, real21, imag21)
    
    test_eval = [
        "dB(S(1,1))",
        "min(dB(S(1,1)))",
        "max(dB(S(1,1)))",
        "dB(S(1,1)) + dB(S(2,1))",
    ]
    
    evaluator = FormulaEvaluator(s_data)
    for formula in test_eval:
        result, err = evaluate_formula(formula, s_data)
        if err:
            logger.info(f"  {formula}: 错误 - {err}")
        else:
            if isinstance(result, np.ndarray):
                logger.info(f"  {formula}: array(shape={result.shape}, min={result.min():.2f}, max={result.max():.2f})")
            else:
                logger.info(f"  {formula}: {result:.4f}")