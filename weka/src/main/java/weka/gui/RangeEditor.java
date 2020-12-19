/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 *    RangeEditor.java
 *    Copyright (C) 2020 University of Waikato, Hamilton, New Zealand
 *
 */
package weka.gui;

import weka.core.Range;
import java.beans.PropertyEditorSupport;

/** 
 * A PropertyEditor that can be used to edit Range objects (really, just appropriately formatted strings).
 *
 * @author Eibe Frank (eibe@waikato.ac.nz)
 * @version $Revision: 8034 $
 */
public class RangeEditor extends PropertyEditorSupport {

  /**
   * Returns a description of the property value as java source.
   *
   * @return a value of type 'String'
   */
  public String getJavaInitializationString() {
    return "new Range(" + getAsText() + ")";
  }

  /**
   * Gets the current value as text.
   *
   * @return a value of type 'String'
   */
  public String getAsText() {
    return ((Range) getValue()).getRanges();
  }

  /**
   * Sets the current property value as text.
   *
   * @param text the text of the selected tag.
   * @throws IllegalArgumentException if an error occurs
   */
  public void setAsText(String text) {
    try {
      setValue(new Range(text));
    } catch (Exception ex) {
      throw new IllegalArgumentException(text);
    }
  }
}

