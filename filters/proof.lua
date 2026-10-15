local function has_class(el, class_name)
  for _, class in ipairs(el.classes) do
    if class == class_name then
      return true
    end
  end
  return false
end

local function add_class(el, class_name)
  if not has_class(el, class_name) then
    table.insert(el.classes, class_name)
  end
end

local function latex_escape(text)
  return text
    :gsub("\\", "\\textbackslash{}")
    :gsub("([%%$#&_{}])", "\\%1")
    :gsub("~", "\\textasciitilde{}")
    :gsub("%^", "\\textasciicircum{}")
end

function Div(el)
  local is_proof = has_class(el, "proof")
  local is_example = has_class(el, "example") or el.identifier == "example"

  if is_example then
    add_class(el, "example")
  end

  if FORMAT ~= "latex" and FORMAT ~= "pdf" then
    if is_example then
      return el
    end
    return nil
  end

  if is_proof then
    local title = el.attributes.title
    local begin = "\\begin{proof}"
    if title ~= nil and title ~= "" then
      begin = "\\begin{proof}[" .. latex_escape(title) .. "]"
    end

    table.insert(el.content, 1, pandoc.RawBlock("latex", begin))
    table.insert(el.content, pandoc.RawBlock("latex", "\\end{proof}"))
    return el.content
  end

  if is_example then
    table.insert(el.content, 1, pandoc.RawBlock("latex", "\\begin{example}"))
    table.insert(el.content, pandoc.RawBlock("latex", "\\end{example}"))
    return el.content
  end

  return nil
end
