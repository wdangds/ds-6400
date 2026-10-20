local function has_class(el, class_name)
  for _, class in ipairs(el.classes) do
    if class == class_name then
      return true
    end
  end
  return false
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

  if FORMAT ~= "latex" and FORMAT ~= "pdf" then
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

  return nil
end
