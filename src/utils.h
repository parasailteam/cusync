#include<string>

#pragma once

static std::string indentStr(int level) {
  const std::string perLevel = "  ";
  std::string o = "";
  for (int i = 0; i < level; i++) {
    o += perLevel;
  }
  return o;
}